"""
run_libero_remote.py  (host에서 실행 — LIBERO 시뮬)
--------------------------------------------------
LIBERO closed-loop rollout을 돌리되, 정책 추론은 Jetson 서버(RemotePolicy)에 위임한다.
→ 시뮬은 host, 모델은 Jetson.

이번 개정 반영 사항 (설계서 §2·3·4 + 확정 결정)
  - suite별 성공률 분해: 여러 suite를 한 번에 순회해 결과를 나란히 CSV에 쌓는다.
  - action smoothness(LDLJ+JerkRMS): 성공 episode의 action 시퀀스만 계산.
  - 제어주파수-성공률 곡선: --target-hz 로 인위적 지연을 주입해 스윕.
  - success rate 통계: --seeds 로 3회(기본) 반복 후 평균/표준편차/95% CI 보고.

실행 예)
  # 기본(단일 suite, 3 seed, 지연 없음)
  python -m serving.run_libero_remote --server 192.168.0.42:5555 \
      --artifact A2_int4 --suites libero_spatial --episodes 50 --seeds 0 1 2

  # 4개 suite 전부 + target_hz 스윕
  python -m serving.run_libero_remote --server 192.168.0.42:5555 \
      --artifact A2_int4 --suites libero_spatial libero_object libero_goal libero_10 \
      --episodes 50 --seeds 0 1 2 --target-hz-sweep

주의
- LIBERO 성공 판정(연속 10 step 유지)의 엄밀한 재현은 공식 harness를 따르는 것이 안전하다.
  이 파일은 원격 정책을 잇는 최소 골격이며, 논문용 수치는 공식 run_libero_eval.py의
  정책 호출부를 RemotePolicy로 교체하는 방식을 권장한다.
- seed는 로컬(에피소드 순서/서브샘플링)의 확률성만 제어한다. 정책 자체의 stochasticity는
  서버(Jetson) 쪽 설정에 따른다(do_sample=False 권장 — 재현성 확보).
"""

import argparse
import os
import time
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from serving.policy_client import RemotePolicy
from bench_common import append_csv_row, agg_seed_runs
from smoothness import compute_smoothness

# suite별 대략적 max step (공식값에 맞춰 조정)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 220,
             "libero_goal": 300, "libero_10": 520}

ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TARGET_HZ_SWEEP = [30, 15, 10, 6, 3]  # 확정된 제안값


def make_env(suite_name, task_id, res=256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
    init_states = suite.get_task_init_states(task_id)
    instruction = getattr(task, "language", "complete the task")
    return env, init_states, instruction


def obs_to_dict(obs, instruction):
    """LIBERO 관측 → 서버가 이해하는 obs dict. 서버가 모델별로 재가공한다."""
    def _img(key):
        v = obs.get(key)
        return None if v is None else np.asarray(v, dtype=np.uint8)
    state = np.concatenate([
        np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), np.float32),
        np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), np.float32),
        np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), np.float32),
    ])
    return {
        "image": _img("agentview_image"),
        "image2": _img("robot0_eye_in_hand_image"),
        "state": state,
        "instruction": instruction,
    }


def _safe_filename(text: str, maxlen: int = 40) -> str:
    """instruction 문자열을 파일명으로 써도 안전하게 정리한다(공백/특수문자 처리)."""
    keep = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)
    keep = "_".join(keep.split())
    return (keep[:maxlen] or "task")


def save_video(frames, path: str, fps: int = 10) -> bool:
    """
    프레임 리스트를 mp4로 저장한다. imageio가 없으면 건너뛰고 설치 안내만 출력한다
    (영상 저장 실패로 측정 자체가 죽으면 안 되므로 예외를 여기서 흡수한다).
    """
    if not frames:
        return False
    try:
        import imageio
    except ImportError:
        print("[record] imageio가 설치돼 있지 않아 영상 저장을 건너뜀. "
              "설치: pip install imageio imageio-ffmpeg")
        return False
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.mimsave(path, frames, fps=fps)
        return True
    except Exception as e:
        print(f"[record] 영상 저장 실패({path}): {e}")
        return False


def _write_live_view(path: str, image: np.ndarray, text_path: str = None, text: str = None) -> None:
    """
    현재 프레임을 하나의 이미지 파일로 계속 덮어쓴다(진짜 동영상 창이 아니라,
    "새로고침하면 최신 화면이 보이는" 방식 — headless(EGL) 환경에서 유일하게
    안정적으로 되는 실시간 확인 방법이다).
    text/text_path를 주면 지금 어떤 task를 수행 중인지 별도 텍스트 파일로도 남긴다.
    """
    try:
        import imageio
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.imwrite(path, image)
        if text_path and text is not None:
            with open(text_path, "w") as f:
                f.write(text)
    except Exception:
        pass  # 실시간 뷰는 실패해도 측정 자체를 막으면 안 된다


def run_episode(policy, env, init_state, instruction, max_steps, target_hz=None,
                progress_prefix="", step_log_every=20, record=False,
                live_view_path=None, live_view_every=1):
    """
    한 episode를 closed-loop으로 실행한다.
    target_hz가 주어지면 매 step 후 (1/target_hz - 실제소요) 만큼 sleep해 제어주파수를 강제한다.
    반환: (success: bool, action_seq: np.ndarray [T, action_dim], frames: list|None)

    [신규] record=True면 매 step의 agentview_image를 frames 리스트에 모아 반환한다.
    실제로 파일로 저장할지는 호출자(run_suite_once)가 정책(주기적/실패시 등)에 따라
    결정한다 — 여기서는 "모으기만" 하고 "저장 여부 판단"은 분리했다.

    [신규] live_view_path를 주면, live_view_every step마다 현재 화면을 그 경로에
    계속 덮어쓴다("live_view.jpg"를 이미지 뷰어로 열어두고 자동새로고침 설정하면
    실시간에 가깝게 볼 수 있다). record(영상 파일 저장)와는 목적이 다르다:
    - record: 나중에 다시 보기 위해 저장(선택적, 주기/실패시)
    - live_view: 지금 이 순간 진행 상황을 눈으로 확인하기 위한 것(항상 켜도 오버헤드 작음)

    [신규] progress_prefix/step_log_every: 진행 상황을 step 단위로도 출력한다.
    - 긴 episode(최대 520 step인 suite도 있음) 도중 화면이 몇 분씩 조용해지는 걸 막는다.
    - 만약 policy.predict()가 예외(예: zmq timeout)를 던지면, "정확히 몇 번째 step에서
      멈췄는지"를 출력하고 다시 던진다 — 이후 같은 문제가 재발해도 원인 위치를 바로
      알 수 있게 하기 위함이다(이전에 겪은 timeout 사고에서 이 정보가 없어 어려웠다).

    주의(기존): 사설 메서드 env._get_observations()를 쓰지 않는다. LIBERO/robosuite
    버전에 따라 존재 여부·래핑 구조(env vs env.env)가 달라 깨지기 쉽기 때문이다.
    대신 공개 API인 reset()/set_init_state()/step()의 "반환값"만으로 obs를 이어간다.
    """
    reset_obs = env.reset()
    init_obs = env.set_init_state(init_state)
    # set_init_state가 obs를 안 돌려주는 버전 대비: reset_obs로 폴백.
    obs = init_obs if isinstance(init_obs, dict) else reset_obs

    done = False
    actions = []
    frames = [] if record else None
    ep_t0 = time.perf_counter()
    live_text_path = (os.path.splitext(live_view_path)[0] + ".txt") if live_view_path else None

    for step_i in range(max_steps):
        if record:
            frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8))

        if live_view_path and (step_i % live_view_every == 0):
            status = (f"{progress_prefix}\ninstruction: {instruction}\n"
                     f"step: {step_i+1}/{max_steps}")
            _write_live_view(live_view_path, obs["agentview_image"], live_text_path, status)

        t0 = time.perf_counter()
        try:
            action = policy.predict(obs_to_dict(obs, instruction))
        except Exception as e:
            elapsed_ep = time.perf_counter() - ep_t0
            print(f"{progress_prefix} [에러] step {step_i+1}/{max_steps}에서 정책 호출 실패 "
                  f"(episode 경과 {elapsed_ep:.1f}s): {e}")
            raise
        elapsed = time.perf_counter() - t0

        actions.append(np.asarray(action, dtype=np.float32))
        obs, reward, done, info = env.step(action.tolist())  # 다음 루프의 obs를 여기서 확보

        if target_hz:
            budget = 1.0 / target_hz
            if elapsed < budget:
                time.sleep(budget - elapsed)

        if step_log_every and (step_i + 1) % step_log_every == 0 and not done:
            print(f"{progress_prefix}   step {step_i+1}/{max_steps} 진행 중 "
                  f"(episode 경과 {time.perf_counter()-ep_t0:.1f}s, "
                  f"최근 predict {elapsed*1000:.0f}ms)")

        if done:
            break

    action_seq = np.stack(actions) if actions else np.zeros((0, 7), np.float32)
    return bool(done), action_seq, frames


def run_suite_once(policy, suite_name, n_tasks, episodes, seed, out_csv,
                   artifact, target_hz=None, step_log_every=20,
                   record_every=0, record_failures=False, record_dir="./benchmark/videos",
                   live_view_path=None, live_view_every=1):
    """
    한 suite를 한 seed로 1회 실행. 성공률 + smoothness(성공 episode 평균)를 반환.
    seed는 에피소드 서브샘플링(초기상태 셔플)에 사용해 실행 간 변동을 만든다.

    [신규] episode 하나가 끝날 때마다 진행 로그를 남긴다(누적 성공률, 총 경과 시간,
    "몇 번째 episode인지"). 기존엔 suite 전체(수십 episode)가 끝나야 첫 출력이 나와
    16분씩 화면이 조용했던 문제를 해결한다.

    [신규] task마다 자연어 instruction을 출력한다 — "task 3/10"처럼 번호로만 아는 게
    아니라, 정확히 어떤 지시문("pick up the black bowl...")인지 그대로 보여준다.

    [신규] record_every>0이면 N번째 episode마다, record_failures=True면 실패한 episode마다
    영상을 저장한다(둘 다 꺼져 있으면 프레임을 아예 안 모아 오버헤드 없음). 파일명에
    artifact·suite·seed·task·instruction·성공여부를 담아, 나중에 "어떤 상황에서 실패했는지"
    파일명만 보고도 알 수 있게 한다.

    [신규] live_view_path를 주면 실행 내내 그 경로의 이미지 파일이 계속 최신 화면으로
    갱신된다(headless 환경에서의 "실시간 보기"). 옆에 같은 이름의 .txt 파일에는 지금
    수행 중인 instruction·진행 step이 함께 기록된다.
    """
    rng = np.random.RandomState(seed)
    max_steps = MAX_STEPS.get(suite_name, 300)
    total, success = 0, 0
    ldlj_list, jerkrms_list = [], []
    suite_t0 = time.perf_counter()
    want_record = bool(record_every) or record_failures  # 이게 꺼져있으면 프레임 수집 자체를 생략

    dt = 1.0 / target_hz if target_hz else 1.0  # smoothness 계산용 시간 스케일

    for task_id in range(n_tasks):
        env, init_states, instruction = make_env(suite_name, task_id)
        print(f"[{suite_name}] task {task_id+1}/{n_tasks} instruction: \"{instruction}\"")
        order = rng.permutation(len(init_states))[:min(episodes, len(init_states))]
        n_this_task = len(order)

        for ep_i, idx in enumerate(order):
            prefix = (f"[{suite_name}][seed={seed}]"
                     f"[task {task_id+1}/{n_tasks}][ep {ep_i+1}/{n_this_task}]")
            ep_t0 = time.perf_counter()

            ok, action_seq, frames = run_episode(
                policy, env, init_states[idx], instruction, max_steps, target_hz,
                progress_prefix=prefix, step_log_every=step_log_every, record=want_record,
                live_view_path=live_view_path, live_view_every=live_view_every)

            ep_dt = time.perf_counter() - ep_t0
            total += 1
            success += int(ok)
            running_rate = 100.0 * success / total
            total_elapsed_min = (time.perf_counter() - suite_t0) / 60.0
            print(f"{prefix} {'✓성공' if ok else '✗실패'} "
                  f"({ep_dt:.1f}s, {len(action_seq)}step) | "
                  f"누적 {success}/{total}({running_rate:.1f}%) | "
                  f"총 경과 {total_elapsed_min:.1f}분")

            # 저장 여부 판단: 주기적 샘플 또는 실패 episode
            save_reason = []
            if record_every and (ep_i + 1) % record_every == 0:
                save_reason.append("periodic")
            if record_failures and not ok:
                save_reason.append("fail")
            if save_reason and frames:
                fname = (f"{artifact}_{suite_name}_seed{seed}_task{task_id}"
                        f"_{_safe_filename(instruction)}_ep{ep_i+1}"
                        f"_{'success' if ok else 'fail'}.mp4")
                path = os.path.join(record_dir, fname)
                if save_video(frames, path):
                    print(f"{prefix} [record] 저장({'+'.join(save_reason)}): {path}")

            if ok and len(action_seq) >= 5:
                sm = compute_smoothness(action_seq, dt=dt, success=True)
                if sm["smoothness_ldlj"] != "NA":
                    ldlj_list.append(sm["smoothness_ldlj"])
                if sm["smoothness_jerkrms"] != "NA":
                    jerkrms_list.append(sm["smoothness_jerkrms"])
        env.close()

    rate = 100.0 * success / max(total, 1)
    mean_ldlj = float(np.mean(ldlj_list)) if ldlj_list else float("nan")
    mean_jerk = float(np.mean(jerkrms_list)) if jerkrms_list else float("nan")

    row = {"name": artifact, "suite": suite_name, "seed": seed,
           "target_hz": target_hz if target_hz else "NA",
           "episodes": total, "success": success, "success_rate_pct": round(rate, 2),
           "smoothness_ldlj_mean": round(mean_ldlj, 4) if not np.isnan(mean_ldlj) else "NA",
           "smoothness_jerkrms_mean": round(mean_jerk, 6) if not np.isnan(mean_jerk) else "NA",
           "n_success_episodes_for_smoothness": len(ldlj_list),
           **policy.stats()}
    append_csv_row(out_csv, row)
    print(f"[{suite_name}][seed={seed}][hz={target_hz}] "
          f"success={rate:.1f}% ({success}/{total})  LDLJ={mean_ldlj:.3f}  JerkRMS={mean_jerk:.5f}")
    return rate, mean_ldlj, mean_jerk


def run_suite_multiseed(policy, suite_name, n_tasks, episodes, seeds, out_csv,
                        artifact, target_hz=None, summary_csv=None, step_log_every=20,
                        record_every=0, record_failures=False, record_dir="./benchmark/videos",
                        live_view_path=None, live_view_every=1):
    """
    같은 suite를 여러 seed로 반복해 success_rate의 평균/표준편차/95% CI를 낸다(확정: 3 seed 기본).
    """
    rates = []
    for seed in seeds:
        rate, _, _ = run_suite_once(policy, suite_name, n_tasks, episodes, seed,
                                    out_csv, artifact, target_hz, step_log_every,
                                    record_every, record_failures, record_dir,
                                    live_view_path, live_view_every)
        rates.append(rate)

    agg = agg_seed_runs(rates)
    print(f"[{suite_name}][hz={target_hz}] === {agg['n_seeds']}-seed 집계 === "
          f"mean={agg['mean']}%  std={agg['std']}  95%CI=[{agg['ci95_low']}, {agg['ci95_high']}]")

    if summary_csv:
        row = {"name": artifact, "suite": suite_name,
               "target_hz": target_hz if target_hz else "NA",
               "n_seeds": agg["n_seeds"], "mean_success_pct": agg["mean"],
               "std": agg["std"], "ci95_low": agg["ci95_low"], "ci95_high": agg["ci95_high"],
               "seed_values": str(agg["values"])}
        append_csv_row(summary_csv, row)
    return agg


def run_all_suites(policy, suites, n_tasks, episodes, seeds, out_csv, summary_csv,
                   artifact, target_hz=None, step_log_every=20,
                   record_every=0, record_failures=False, record_dir="./benchmark/videos",
                   live_view_path=None, live_view_every=1):
    """설계서 §2: 여러 suite를 순회해 능력별 성공률 분해를 얻는다."""
    results = {}
    for suite_name in suites:
        results[suite_name] = run_suite_multiseed(
            policy, suite_name, n_tasks, episodes, seeds, out_csv,
            artifact, target_hz, summary_csv, step_log_every,
            record_every, record_failures, record_dir,
            live_view_path, live_view_every,
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="host:port (Jetson)")
    ap.add_argument("--suites", nargs="+", default=["libero_spatial"],
                    help=f"기본 단일 suite. 전체는: {ALL_SUITES}")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="확정: 3 seed 기본(0 1 2)")
    ap.add_argument("--artifact", default="unknown", help="측정 대상 artifact 이름(로그용)")
    ap.add_argument("--out-csv", default="./benchmark/success_remote.csv")
    ap.add_argument("--summary-csv", default="./benchmark/success_summary.csv")
    ap.add_argument("--target-hz", type=float, default=None,
                    help="단일 목표 Hz로 지연 주입(제어주파수 강제)")
    ap.add_argument("--target-hz-sweep", action="store_true",
                    help=f"확정 스윕값 {TARGET_HZ_SWEEP}Hz 전체를 순회")
    # [신규] 진행 상황 로그 주기. 0이면 episode 안 step 로그를 끈다(episode 단위 로그는 항상 켜짐).
    ap.add_argument("--step-log-every", type=int, default=20,
                    help="이 step마다 진행 로그 출력(기본 20, 0이면 끔)")
    # [신규] Jetson 응답 대기 시간. 기본 15초는 A0(7B) 등 느린 모델에서 timeout이 날 수 있어 늘림.
    ap.add_argument("--timeout-ms", type=int, default=60000,
                    help="policy 응답 대기 timeout(ms), 기본 60초")
    # [신규] 영상 저장 옵션. 기본은 전부 꺼짐(프레임 수집 자체가 없어 오버헤드 0).
    ap.add_argument("--record-every", type=int, default=0,
                    help="이 episode마다 1개씩 영상 저장(0이면 끔). 예: 10 → 10개마다 1개")
    ap.add_argument("--record-failures", action="store_true",
                    help="실패한 episode는 항상 영상 저장(디버깅용)")
    ap.add_argument("--record-dir", default="./benchmark/videos",
                    help="영상 저장 폴더(기본 ./benchmark/videos)")
    # [신규] 실시간 보기. headless라 진짜 창은 못 띄우지만, 이 경로의 이미지 파일이
    # 매 step 계속 갱신된다 — 이미지 뷰어(예: feh --reload, 또는 파일탐색기 미리보기)로
    # 열어두면 사실상 실시간처럼 볼 수 있다. 옆에 같은 이름 .txt에 현재 instruction도 남는다.
    ap.add_argument("--live-view", default=None,
                    help="이 경로에 현재 화면을 계속 덮어써 저장(예: ./live_view.jpg). "
                         "기본 None(끔)")
    ap.add_argument("--live-view-every", type=int, default=1,
                    help="몇 step마다 live-view를 갱신할지(기본 1=매 step)")
    args = ap.parse_args()

    host, port = args.server.split(":")
    policy = RemotePolicy(host, int(port), timeout_ms=args.timeout_ms)
    print("[client] ping:", policy.ping())

    # [신규] 시작 전 전체 규모를 미리 안내 — "이게 대략 몇 episode짜리 실행인지" 가늠하게 한다.
    total_episodes = len(args.suites) * args.n_tasks * args.episodes * len(args.seeds)
    hz_note = f" x {len(TARGET_HZ_SWEEP)}개 target_hz" if args.target_hz_sweep else ""
    print(f"[계획] suites={args.suites} x n_tasks={args.n_tasks} x episodes={args.episodes} "
          f"x seeds={args.seeds}{hz_note} → 총 약 {total_episodes}{hz_note and '×N'} episode 예정")
    if args.record_every or args.record_failures:
        print(f"[계획] 영상 저장 활성화: every={args.record_every}, "
              f"failures={args.record_failures}, dir={args.record_dir}")
    if args.live_view:
        print(f"[계획] 실시간 보기 활성화: {args.live_view} "
              f"(같은 이름 .txt에 instruction 함께 기록, {args.live_view_every} step마다 갱신)")

    if args.target_hz_sweep:
        for hz in TARGET_HZ_SWEEP:
            print(f"\n########## target_hz={hz} ##########")
            run_all_suites(policy, args.suites, args.n_tasks, args.episodes, args.seeds,
                           args.out_csv, args.summary_csv, args.artifact, target_hz=hz,
                           step_log_every=args.step_log_every,
                           record_every=args.record_every, record_failures=args.record_failures,
                           record_dir=args.record_dir,
                           live_view_path=args.live_view, live_view_every=args.live_view_every)
    else:
        run_all_suites(policy, args.suites, args.n_tasks, args.episodes, args.seeds,
                       args.out_csv, args.summary_csv, args.artifact, target_hz=args.target_hz,
                       step_log_every=args.step_log_every,
                       record_every=args.record_every, record_failures=args.record_failures,
                       record_dir=args.record_dir,
                       live_view_path=args.live_view, live_view_every=args.live_view_every)

    policy.close()


if __name__ == "__main__":
    main()
