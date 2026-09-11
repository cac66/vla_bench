"""
policy_server.py  (Jetson에서 실행)
-----------------------------------
선택한 artifact(모델)를 로드하고, host의 시뮬로부터 관측을 받아 action을 돌려준다.
→ "모델 추론은 Jetson에서" 원칙을 코드로 강제하는 핵심 파일.

실행 예)
  # AGX Orin에서 OpenVLA INT4 서빙
  python -m serving.policy_server --model openvla --artifact A2_int4 --port 5555
  # Orin Nano에서 SmolVLA step 감축본 서빙 + 호출별 에너지 측정
  python -m serving.policy_server --model smolvla --artifact B1_step3 --port 5555 --measure-energy
"""

import argparse
import time

from serving.protocol import make_server_socket, send_obj, recv_obj
from energy import SysfsPowerSampler, measure_idle_baseline


def build_servable(model_family: str, artifact: str):
    """model family에 맞는 loader + predict_fn을 반환한다(bench 모듈 재사용)."""
    if model_family == "openvla":
        import bench_openvla as M
        predict = M.predict_openvla
    elif model_family == "smolvla":
        import bench_smolvla as M
        predict = M.predict_smolvla
    else:
        raise ValueError(model_family)

    if artifact not in M.LOADERS:
        raise KeyError(f"{artifact} 없음. 가능: {list(M.LOADERS)}")
    model, extras = M.LOADERS[artifact]()
    return model, extras, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["openvla", "smolvla"])
    ap.add_argument("--artifact", required=True, help="예: A2_int4 / B1_step3")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--measure-energy", action="store_true",
                    help="predict() 호출마다 sysfs로 에너지를 직접 재 infer_energy_j로 응답에 실어보낸다")
    ap.add_argument("--idle-baseline-s", type=float, default=30.0,
                    help="세션 시작 시 1회 측정할 idle-baseline 구간(초). 확정값 30초")
    args = ap.parse_args()

    # idle-baseline은 세션당 1회만 측정해 재사용한다(매 predict()마다 재측정하면 비효율적이고,
    # 서버 프로세스 관점에서 idle 전력은 세션 내내 크게 변하지 않는다고 가정).
    idle_baseline_w = 0.0
    if args.measure_energy:
        idle_baseline_w = measure_idle_baseline(duration_s=args.idle_baseline_s)

    print(f"[server] {args.model}/{args.artifact} 로딩 …")
    model, extras, predict = build_servable(args.model, args.artifact)
    sock = make_server_socket(args.port)
    print(f"[server] 준비 완료. tcp://0.0.0.0:{args.port} 대기")

    while True:
        obs = recv_obj(sock)                 # host로부터 관측 수신
        if isinstance(obs, dict) and obs.get("__cmd__") == "ping":
            send_obj(sock, {"pong": True, "model": args.model, "artifact": args.artifact})
            continue

        if args.measure_energy:
            # predict() 구간만 개별로 감싸 호출별 에너지를 응답에 실어보낸다 — episode/step
            # 단위로 에너지가 바로 귀속되어 host 쪽(run_libero_remote.py) 집계가 쉬워진다.
            with SysfsPowerSampler(interval_s=0.005) as sampler:
                t0 = time.perf_counter()
                action = predict(model, extras, obs)  # ★ Jetson에서 추론
                infer_ms = (time.perf_counter() - t0) * 1000
            net_power_w = max(sampler.mean_power_w() - idle_baseline_w, 0.0)
            infer_energy_j = net_power_w * sampler.duration_s()
        else:
            t0 = time.perf_counter()
            action = predict(model, extras, obs)  # ★ Jetson에서 추론
            infer_ms = (time.perf_counter() - t0) * 1000
            infer_energy_j = None

        send_obj(sock, {"action": action, "infer_ms": infer_ms, "infer_energy_j": infer_energy_j})


if __name__ == "__main__":
    main()
