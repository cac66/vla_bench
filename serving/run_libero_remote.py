"""
run_libero_remote.py  (host에서 실행 — LIBERO 시뮬)
--------------------------------------------------
LIBERO closed-loop rollout을 돌리되, 정책 추론은 Jetson 서버(RemotePolicy)에 위임한다.
→ 시뮬은 host, 모델은 Jetson.

이번 개정 반영 사항 (설계서 §2·3·4 + 확정 결정)
  - suite별 성공률 분해: 여러 suite를 한 번에 순회해 결과를 나란히 CSV에 쌓는다.
  - action smoothness(LDLJ+JerkRMS): 성공 episode의 action 시퀀스만 계산.
  - 제어주파수-성공률 곡선: --target-hz 로 인위적 지연을 주입해 스윕.
    → raw(실측 RTT 기준)와 adjusted(서버 infer_ms만 기준, 네트워크 시간 제외) 두 곡선을
      동시에 산출한다. RemotePolicy가 왕복(net)과 서버측 순수 추론(infer)을 분리 로깅하므로
      가능해졌다 — 자세한 배경은 run_episode()/run_suite_once() docstring 참조.
  - success rate 통계: --seeds 로 3회(기본) 반복 후 평균/표준편차/95% CI 보고(raw/adjusted 각각).
  - episode 직접실측 에너지: 서버가 --measure-energy로 predict() 호출별 에너지를 함께 보내면,
    성공 episode들의 총 에너지 평균을 energy_per_success_j_direct로 CSV에 남긴다.

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


def run_episode(policy, env, init_state, instruction, max_steps, target_hz=None,
                delay_basis="rtt"):
    """
    한 episode를 closed-loop으로 실행한다.
    target_hz가 주어지면 매 step 후 지연을 주입해 제어주파수를 강제하되,
    무엇을 "이 step이 예산을 넘겼는가"의 기준으로 쓸지는 delay_basis가 결정한다.
      - "rtt"  : 실측 왕복시간(policy.predict() 호출의 네트워크+추론) 기준 — 곡선 A(raw).
                 실제 decoupled 배포에서 체감하는 제어주파수를 재현한다.
      - "infer": 서버가 보고한 순수 추론시간(policy.last_infer_ms) 기준 — 곡선 B(adjusted).
                 네트워크 왕복을 뺀, "순수 on-device였다면"을 재해석한 판정이다.
    반환: dict(success: bool, action_seq: np.ndarray [T, action_dim], energy_j: float|None)
      energy_j는 이 episode 동안 서버가 실측한 에너지 총합(predict() 호출별 infer_energy_j의 합).
      서버가 에너지 측정을 지원하지 않으면(--measure-energy 미지정) None.

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
    energy_total = 0.0
    energy_recorded = False

    for _ in range(max_steps):
        t0 = time.perf_counter()
        action = policy.predict(obs_to_dict(obs, instruction))
        elapsed = time.perf_counter() - t0

        e = policy.last_energy_j
        if e is not None:
            energy_total += e
            energy_recorded = True

        actions.append(np.asarray(action, dtype=np.float32))
        obs, reward, done, info = env.step(action.tolist())  # 다음 루프의 obs를 여기서 확보

        if target_hz:
            budget = 1.0 / target_hz
            basis_elapsed = elapsed if delay_basis == "rtt" else (policy.last_infer_ms / 1000.0)
            if basis_elapsed < budget:
                time.sleep(budget - basis_elapsed)

        if done:
            break

    action_seq = np.stack(actions) if actions else np.zeros((0, 7), np.float32)
    return {"success": bool(done), "action_seq": action_seq,
            "energy_j": energy_total if energy_recorded else None}


def run_suite_once(policy, suite_name, n_tasks, episodes, seed, out_csv,
                   artifact, target_hz=None):
    """
    한 suite를 한 seed로 1회 실행. 성공률(raw/adjusted 두 계열) + smoothness(성공 episode 평균)
    + episode 직접실측 에너지를 반환한다.
    seed는 에피소드 서브샘플링(초기상태 셔플)에 사용해 실행 간 변동을 만든다.

    raw vs adjusted (구현 방식: 독립 rollout — 정확도 우선):
      target_hz가 주어지면 같은 (task, init_state)를 raw 기준(elapsed=RTT)과 adjusted
      기준(elapsed=서버 infer_ms) 각각으로 "독립적으로" 두 번 굴린다. 지연 주입 위치가
      다르면 이후 물리 시뮬레이션 진행이 달라져(action이 늦게 나갈수록 env가 그만큼
      "밀린" 상태에서 다음 obs를 받음) 같은 rollout에서 두 판정을 동시에 내는 방식은
      정확하지 않기 때문이다. target_hz가 없으면(지연 주입 자체가 없으므로) 두 기준이
      물리적으로 동일한 rollout이 되어 raw 결과를 그대로 재사용하고 두 번째 rollout은
      건너뛴다(불필요한 비용 절감).
    """
    rng = np.random.RandomState(seed)
    max_steps = MAX_STEPS.get(suite_name, 300)
    total, success_raw, success_adjusted = 0, 0, 0
    ldlj_list, jerkrms_list = [], []
    success_energy_list = []  # raw rollout 기준, 성공 episode들의 실측 총 에너지(J)

    dt = 1.0 / target_hz if target_hz else 1.0  # smoothness 계산용 시간 스케일

    for task_id in range(n_tasks):
        env, init_states, instruction = make_env(suite_name, task_id)
        order = rng.permutation(len(init_states))[:min(episodes, len(init_states))]
        for idx in order:
            res_raw = run_episode(policy, env, init_states[idx], instruction,
                                  max_steps, target_hz, delay_basis="rtt")
            total += 1
            success_raw += int(res_raw["success"])
            if res_raw["success"] and len(res_raw["action_seq"]) >= 5:
                sm = compute_smoothness(res_raw["action_seq"], dt=dt, success=True)
                if sm["smoothness_ldlj"] != "NA":
                    ldlj_list.append(sm["smoothness_ldlj"])
                if sm["smoothness_jerkrms"] != "NA":
                    jerkrms_list.append(sm["smoothness_jerkrms"])
            if res_raw["success"] and res_raw["energy_j"] is not None:
                success_energy_list.append(res_raw["energy_j"])

            if target_hz:
                res_adj = run_episode(policy, env, init_states[idx], instruction,
                                      max_steps, target_hz, delay_basis="infer")
                success_adjusted += int(res_adj["success"])
            else:
                success_adjusted += int(res_raw["success"])
        env.close()

    rate_raw = 100.0 * success_raw / max(total, 1)
    rate_adjusted = 100.0 * success_adjusted / max(total, 1)
    mean_ldlj = float(np.mean(ldlj_list)) if ldlj_list else float("nan")
    mean_jerk = float(np.mean(jerkrms_list)) if jerkrms_list else float("nan")
    energy_per_success_j_direct = (
        round(float(np.mean(success_energy_list)), 4) if success_energy_list else "NA"
    )

    stats = policy.stats()
    network_overhead_ms_mean = (
        stats["rtt_ms_mean"] - stats["server_infer_ms_mean"]
        if not (np.isnan(stats["rtt_ms_mean"]) or np.isnan(stats["server_infer_ms_mean"]))
        else float("nan")
    )

    row = {"name": artifact, "suite": suite_name, "seed": seed,
           "target_hz": target_hz if target_hz else "NA",
           "episodes": total,
           "success": success_raw, "success_rate_pct": round(rate_raw, 2),  # 하위 호환 alias(raw)
           "success_raw": success_raw, "success_rate_pct_raw": round(rate_raw, 2),
           "success_adjusted": success_adjusted,
           "success_rate_pct_adjusted": round(rate_adjusted, 2),
           "smoothness_ldlj_mean": round(mean_ldlj, 4) if not np.isnan(mean_ldlj) else "NA",
           "smoothness_jerkrms_mean": round(mean_jerk, 6) if not np.isnan(mean_jerk) else "NA",
           "n_success_episodes_for_smoothness": len(ldlj_list),
           "energy_per_success_j_direct": energy_per_success_j_direct,
           "n_success_episodes_for_energy": len(success_energy_list),
           "network_overhead_ms_mean": round(network_overhead_ms_mean, 4)
                                        if not np.isnan(network_overhead_ms_mean) else "NA",
           **stats}
    append_csv_row(out_csv, row)
    print(f"[{suite_name}][seed={seed}][hz={target_hz}] "
          f"success_raw={rate_raw:.1f}% success_adjusted={rate_adjusted:.1f}% "
          f"({success_raw}/{total} vs {success_adjusted}/{total})  "
          f"LDLJ={mean_ldlj:.3f}  JerkRMS={mean_jerk:.5f}")
    return {"rate_raw": rate_raw, "rate_adjusted": rate_adjusted,
           "mean_ldlj": mean_ldlj, "mean_jerk": mean_jerk,
           "energy_per_success_j_direct": energy_per_success_j_direct,
           # merge_results.py의 latency_consistency_check가 트랙2 latency를 읽어올 수 있도록
           # summary_csv에도 실어보낸다(out_csv에만 있으면 --success 기본값인 success_summary.csv
           # 기준 merge에서는 항상 데이터가 없어 "NA"만 나오게 된다).
           "server_infer_ms_mean": stats["server_infer_ms_mean"],
           "rtt_ms_mean": stats["rtt_ms_mean"]}


def run_suite_multiseed(policy, suite_name, n_tasks, episodes, seeds, out_csv,
                        artifact, target_hz=None, summary_csv=None):
    """
    같은 suite를 여러 seed로 반복해 success_rate(raw/adjusted 각각)의
    평균/표준편차/95% CI를 낸다(확정: 3 seed 기본).
    """
    rates_raw, rates_adjusted, energy_per_success_vals = [], [], []
    infer_ms_vals, rtt_ms_vals = [], []
    for seed in seeds:
        res = run_suite_once(policy, suite_name, n_tasks, episodes, seed,
                             out_csv, artifact, target_hz)
        rates_raw.append(res["rate_raw"])
        rates_adjusted.append(res["rate_adjusted"])
        if res["energy_per_success_j_direct"] != "NA":
            energy_per_success_vals.append(res["energy_per_success_j_direct"])
        if not np.isnan(res["server_infer_ms_mean"]):
            infer_ms_vals.append(res["server_infer_ms_mean"])
        if not np.isnan(res["rtt_ms_mean"]):
            rtt_ms_vals.append(res["rtt_ms_mean"])

    agg_raw = agg_seed_runs(rates_raw)
    agg_adjusted = agg_seed_runs(rates_adjusted)
    print(f"[{suite_name}][hz={target_hz}] === {agg_raw['n_seeds']}-seed 집계 === "
          f"raw: mean={agg_raw['mean']}% std={agg_raw['std']} "
          f"95%CI=[{agg_raw['ci95_low']}, {agg_raw['ci95_high']}]  |  "
          f"adjusted: mean={agg_adjusted['mean']}% std={agg_adjusted['std']} "
          f"95%CI=[{agg_adjusted['ci95_low']}, {agg_adjusted['ci95_high']}]")

    if summary_csv:
        row = {"name": artifact, "suite": suite_name,
               "target_hz": target_hz if target_hz else "NA",
               "n_seeds": agg_raw["n_seeds"],
               # 하위 호환 alias: 기존 소비자(merge_results.py 등)는 raw 값을 그대로 읽는다.
               "mean_success_pct": agg_raw["mean"], "std": agg_raw["std"],
               "ci95_low": agg_raw["ci95_low"], "ci95_high": agg_raw["ci95_high"],
               "seed_values": str(agg_raw["values"]),
               "mean_success_pct_raw": agg_raw["mean"], "std_raw": agg_raw["std"],
               "ci95_low_raw": agg_raw["ci95_low"], "ci95_high_raw": agg_raw["ci95_high"],
               "seed_values_raw": str(agg_raw["values"]),
               "mean_success_pct_adjusted": agg_adjusted["mean"], "std_adjusted": agg_adjusted["std"],
               "ci95_low_adjusted": agg_adjusted["ci95_low"], "ci95_high_adjusted": agg_adjusted["ci95_high"],
               "seed_values_adjusted": str(agg_adjusted["values"]),
               # 트랙2 직접실측 energy_per_success(J). seed마다 에너지 측정이 있었을 때만 평균한다
               # — merge_results.py가 트랙1 근사치(energy_per_success_j)와 나란히 비교하는 데 쓴다.
               "energy_per_success_j_direct": (
                   round(float(np.mean(energy_per_success_vals)), 4)
                   if energy_per_success_vals else "NA"
               ),
               # merge_results.py의 latency_consistency_check(트랙1 vs 트랙2)가 읽는 컬럼.
               "server_infer_ms_mean": (
                   round(float(np.mean(infer_ms_vals)), 4) if infer_ms_vals else "NA"
               ),
               "rtt_ms_mean": round(float(np.mean(rtt_ms_vals)), 4) if rtt_ms_vals else "NA"}
        append_csv_row(summary_csv, row)
    return {"raw": agg_raw, "adjusted": agg_adjusted}


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
