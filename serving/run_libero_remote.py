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


def run_episode(policy, env, init_state, instruction, max_steps, target_hz=None):
    """
    한 episode를 closed-loop으로 실행한다.
    target_hz가 주어지면 매 step 후 (1/target_hz - 실제소요) 만큼 sleep해 제어주파수를 강제한다.
    반환: (success: bool, action_seq: np.ndarray [T, action_dim])

    주의(수정): 사설 메서드 env._get_observations()를 쓰지 않는다. LIBERO/robosuite
    버전에 따라 존재 여부·래핑 구조(env vs env.env)가 달라 깨지기 쉽기 때문이다.
    대신 공개 API인 reset()/set_init_state()/step()의 "반환값"만으로 obs를 이어간다
    — 스모크 테스트에서 이미 env.step()이 완전한 obs dict를 준다는 걸 확인했다.
    """
    reset_obs = env.reset()
    init_obs = env.set_init_state(init_state)
    # set_init_state가 obs를 안 돌려주는 버전 대비: reset_obs로 폴백.
    obs = init_obs if isinstance(init_obs, dict) else reset_obs

    done = False
    actions = []

    for _ in range(max_steps):
        t0 = time.perf_counter()
        action = policy.predict(obs_to_dict(obs, instruction))
        elapsed = time.perf_counter() - t0

        actions.append(np.asarray(action, dtype=np.float32))
        obs, reward, done, info = env.step(action.tolist())  # 다음 루프의 obs를 여기서 확보

        if target_hz:
            budget = 1.0 / target_hz
            if elapsed < budget:
                time.sleep(budget - elapsed)

        if done:
            break

    action_seq = np.stack(actions) if actions else np.zeros((0, 7), np.float32)
    return bool(done), action_seq


def run_suite_once(policy, suite_name, n_tasks, episodes, seed, out_csv,
                   artifact, target_hz=None):
    """
    한 suite를 한 seed로 1회 실행. 성공률 + smoothness(성공 episode 평균)를 반환.
    seed는 에피소드 서브샘플링(초기상태 셔플)에 사용해 실행 간 변동을 만든다.
    """
    rng = np.random.RandomState(seed)
    max_steps = MAX_STEPS.get(suite_name, 300)
    total, success = 0, 0
    ldlj_list, jerkrms_list = [], []

    dt = 1.0 / target_hz if target_hz else 1.0  # smoothness 계산용 시간 스케일

    for task_id in range(n_tasks):
        env, init_states, instruction = make_env(suite_name, task_id)
        order = rng.permutation(len(init_states))[:min(episodes, len(init_states))]
        for idx in order:
            ok, action_seq = run_episode(policy, env, init_states[idx], instruction,
                                         max_steps, target_hz)
            total += 1
            success += int(ok)
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
                        artifact, target_hz=None, summary_csv=None):
    """
    같은 suite를 여러 seed로 반복해 success_rate의 평균/표준편차/95% CI를 낸다(확정: 3 seed 기본).
    """
    rates = []
    for seed in seeds:
        rate, _, _ = run_suite_once(policy, suite_name, n_tasks, episodes, seed,
                                    out_csv, artifact, target_hz)
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
                   artifact, target_hz=None):
    """설계서 §2: 여러 suite를 순회해 능력별 성공률 분해를 얻는다."""
    results = {}
    for suite_name in suites:
        results[suite_name] = run_suite_multiseed(
            policy, suite_name, n_tasks, episodes, seeds, out_csv,
            artifact, target_hz, summary_csv,
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
    args = ap.parse_args()

    host, port = args.server.split(":")
    policy = RemotePolicy(host, int(port))
    print("[client] ping:", policy.ping())

    if args.target_hz_sweep:
        for hz in TARGET_HZ_SWEEP:
            print(f"\n########## target_hz={hz} ##########")
            run_all_suites(policy, args.suites, args.n_tasks, args.episodes, args.seeds,
                           args.out_csv, args.summary_csv, args.artifact, target_hz=hz)
    else:
        run_all_suites(policy, args.suites, args.n_tasks, args.episodes, args.seeds,
                       args.out_csv, args.summary_csv, args.artifact, target_hz=args.target_hz)

    policy.close()


if __name__ == "__main__":
    main()
