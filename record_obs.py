"""
record_obs.py  (host에서 실행 — LIBERO 시뮬)
--------------------------------------------
LIBERO 시뮬에서 rollout하며 관측을 녹화해 openvla_eval.npz를 만든다.
별도 다운로드 없이, MSE 스크리닝(bench_openvla.py --mode mse)에 바로 쓸 수 있는
포맷(bench_openvla.load_eval_obs_openvla와 호환)으로 저장한다.

핵심 설계
- action은 0(정지)으로 rollout한다 — 이 스크립트의 목적은 "다양한 장면의 관측 이미지"를
  모으는 것이지 성공적인 시연을 만드는 게 아니다. 그래서 action의 질은 상관없다.
- ground-truth action은 기본적으로 저장하지 않는다. self-reference MSE(FP16 대비 편차)는
  GT가 필요 없고, GT 없이도 스크리닝이 성립하도록 설계했기 때문이다(--with-actions로 켤 수 있음).
- obs 취득은 run_libero_remote.py에서 확정한 안전한 방식을 그대로 따른다: 사설 메서드
  (env._get_observations 등)를 쓰지 않고, reset()/set_init_state()/step()의 반환값만 쓴다.

실행 예)
  python3 tools/record_obs.py --suite libero_spatial --n-tasks 3 --steps-per-task 70 \
      --out data/openvla_eval.npz
"""

import argparse
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


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


def record_task(suite_name, task_id, steps, action_dim=7, with_actions=False):
    """한 task를 rollout하며 (image, instruction, [action]) 리스트를 모은다."""
    env, init_states, instruction = make_env(suite_name, task_id)

    # 안전한 obs 취득: reset()/set_init_state()의 반환값만 사용(사설 메서드 미사용).
    reset_obs = env.reset()
    init_obs = env.set_init_state(init_states[0])
    obs = init_obs if isinstance(init_obs, dict) else reset_obs

    imgs, instrs, actions = [], [], []
    zero_action = [0.0] * action_dim

    for _ in range(steps):
        imgs.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
        instrs.append(instruction)
        if with_actions:
            actions.append(np.asarray(zero_action, dtype=np.float32))
        obs, reward, done, info = env.step(zero_action)
        if done:
            break

    env.close()
    return imgs, instrs, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial",
                    help="libero_spatial / libero_object / libero_goal / libero_10")
    ap.add_argument("--n-tasks", type=int, default=3, help="이 suite에서 몇 개 task를 녹화할지")
    ap.add_argument("--steps-per-task", type=int, default=70, help="task당 몇 step 녹화할지")
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--with-actions", action="store_true",
                    help="GT action(전부 0)도 함께 저장. self-reference MSE엔 불필요, 기본 off")
    ap.add_argument("--out", default="data/openvla_eval.npz")
    args = ap.parse_args()

    all_imgs, all_instrs, all_actions = [], [], []
    for tid in range(args.n_tasks):
        print(f"[record_obs] {args.suite} task {tid} 녹화 중 ({args.steps_per_task} step) …")
        imgs, instrs, actions = record_task(
            args.suite, tid, args.steps_per_task, args.action_dim, args.with_actions
        )
        all_imgs.extend(imgs)
        all_instrs.extend(instrs)
        all_actions.extend(actions)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_kwargs = {
        "images": np.array(all_imgs, dtype=np.uint8),
        "instructions": np.array(all_instrs, dtype=object),
    }
    if args.with_actions:
        save_kwargs["actions"] = np.array(all_actions, dtype=np.float32)

    np.savez(args.out, **save_kwargs)
    print(f"[record_obs] 완료 — {len(all_imgs)}개 관측 → {args.out}")
    if not args.with_actions:
        print("[record_obs] GT action은 저장하지 않았다(self-reference MSE 기본 방식엔 불필요).")


if __name__ == "__main__":
    main()
