"""
tools/profile_compare.py  (AGX Orin 컨테이너 안에서 실행)
------------------------------------------------------------
breakdown_compare.py로 "vision_backbone이 language_model보다 훨씬 크게 느려진다"는
게 확인됐다(A1 vision 6.14x vs backbone 3.20x, A2 vision 2.46x vs backbone 1.57x).
다음 질문은 "그 안에서 정확히 어떤 CUDA 커널이 시간을 먹는가"다.

무엇을 하는가
  - A0/A1/A2 각각의 predict_action() 호출을 torch.profiler로 커널 단위까지 프로파일링한다.
  - 커널별 self_cuda_time_total(그 커널 자체가 쓴 시간, 하위 호출 제외) 기준 상위 N개를 뽑는다.
  - A0에는 없는데 A1/A2에만 새로 나타나는 커널(= 양자화가 추가한 연산, 예: dequantize,
    int8 outlier 분해 등)을 자동으로 골라낸다.
  - A0에도 있던 커널인데 A1/A2에서 유독 비율이 크게 늘어난 것도 함께 뽑는다
    (같은 연산이라도 느려진 이유가 있을 수 있다).

주의
  - 프로파일링 자체가 오버헤드가 크므로, 순수 latency 측정(--mode latency)보다
    반복 횟수를 훨씬 적게(기본 5회) 잡는다. 여기서 나온 절대시간은 latency 측정치와
    직접 비교하지 않는다 — 상대적인 "커널 구성·비중 차이"를 보는 용도다.
  - self_cuda_time_total 기준으로 정렬한다(그 커널 자체의 실행 시간, 하위 호출 제외).
    cuda_time_total(하위 호출 포함, inclusive)이 아니라 이걸 쓰는 이유는, "이 연산
    자체가 무거운가"를 보려는 것이지 "이 연산을 감싼 상위 함수가 무거운가"를 보려는
    게 아니기 때문이다.

실행 예)
  python3 tools/profile_compare.py --artifacts A0_fp16 A1_int8 A2_int4 --n 5 --top 25
"""

import argparse
import os
import sys

import torch
from torch.profiler import profile, ProfilerActivity

# tools/ 안에서 실행해도 상위 폴더(vla_bench/)의 bench_openvla.py를 찾도록 경로 추가.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_openvla as M
from bench_common import free_model


def profile_artifact(artifact_name: str, warmup: int = 5, n: int = 5) -> dict:
    """artifact를 로드해 predict_action()을 n회 프로파일링하고,
    커널명 -> self_cuda_time_total(us) dict를 반환한다."""
    print(f"\n{'='*60}\n[profile] {artifact_name}\n{'='*60}")

    if artifact_name not in M.LOADERS:
        print(f"  [error] LOADERS에 없음: {artifact_name}")
        return {}

    model, extras = M.LOADERS[artifact_name]()
    inputs, seq_len, infer_fn = M.build_inputs_openvla(model, extras)

    print(f"  워밍업 {warmup}회 …")
    for _ in range(warmup):
        _ = infer_fn(model, inputs, extras)
    torch.cuda.synchronize()

    print(f"  프로파일링 {n}회 …")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(n):
            _ = infer_fn(model, inputs, extras)
        torch.cuda.synchronize()

    kernel_times = {}
    for evt in prof.key_averages():
        t = getattr(evt, "self_cuda_time_total", 0) or 0
        if t > 0:
            kernel_times[evt.key] = kernel_times.get(evt.key, 0) + t

    free_model(model)
    return kernel_times


def top_n(kernel_times: dict, n: int = 25):
    return sorted(kernel_times.items(), key=lambda kv: kv[1], reverse=True)[:n]


def us_to_ms(us: float) -> float:
    return us / 1000.0


def print_top(name: str, kernel_times: dict, n: int):
    print(f"\n--- {name} 상위 {n}개 커널 (self_cuda_time_total 기준) ---")
    total = sum(kernel_times.values()) or 1
    for kernel, t in top_n(kernel_times, n):
        pct = 100 * t / total
        print(f"  {us_to_ms(t):>10.3f} ms  ({pct:>5.1f}%)  {kernel}")


def compare_new_kernels(base: dict, other: dict, base_name: str, other_name: str, n: int = 15):
    """base(A0)에는 없거나 미미한데 other(A1/A2)에서 새로 크게 나타난 커널."""
    print(f"\n=== [{other_name}]에만 새로 나타난(또는 {base_name}엔 미미했던) 커널 상위 {n}개 ===")
    new_or_grown = []
    for kernel, t_other in other.items():
        t_base = base.get(kernel, 0)
        if t_base == 0:
            new_or_grown.append((kernel, t_other, None))  # base엔 아예 없음
        else:
            ratio = t_other / t_base
            if ratio > 2.0:  # 2배 넘게 커진 것만
                new_or_grown.append((kernel, t_other, ratio))
    new_or_grown.sort(key=lambda x: x[1], reverse=True)
    for kernel, t_other, ratio in new_or_grown[:n]:
        if ratio is None:
            print(f"  {us_to_ms(t_other):>10.3f} ms  [신규]        {kernel}")
        else:
            print(f"  {us_to_ms(t_other):>10.3f} ms  [{ratio:>5.1f}배]  {kernel}")
    if not new_or_grown:
        print("  (신규/2배 이상 증가 커널 없음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", nargs="+", default=["A0_fp16", "A1_int8", "A2_int4"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n", type=int, default=5, help="프로파일링 반복 횟수(오버헤드 크므로 적게)")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    results = {}
    for a in args.artifacts:
        results[a] = profile_artifact(a, warmup=args.warmup, n=args.n)

    # --- artifact별 상위 커널 목록 ---
    for a, kt in results.items():
        if kt:
            print_top(a, kt, args.top)

    # --- baseline(첫 artifact) 대비 신규/급증 커널 ---
    names = list(results.keys())
    if len(names) >= 2 and results[names[0]]:
        base_name = names[0]
        base = results[base_name]
        for other_name in names[1:]:
            if results[other_name]:
                compare_new_kernels(base, results[other_name], base_name, other_name, n=args.top)

    print("\n[해석 가이드]")
    print("  - 커널명에 'dequant'/'quant'/'int8'/'nf4' 등이 보이면 양자화 전용 연산이다.")
    print("    이게 상위권이면 '양자화 자체의 dequantize 비용'이 병목이라는 뜻.")
    print("  - 'memcpy'/'copy'/'to_copy' 계열이 상위권이면 CPU-GPU 또는 dtype 변환")
    print("    데이터 이동이 병목 — outlier 분해 등에서 흔히 보고되는 패턴과 일치한다.")
    print("  - 'aten::linear'/'aten::matmul' 등 일반 행렬곱 커널의 절대시간이 A0보다")
    print("    크게 늘었다면(신규 커널이 아니라 기존 커널이 2배 이상 됐다면), 같은 연산이")
    print("    양자화 경로에서 비효율적인 커널 구현을 타고 있다는 뜻일 수 있다.")


if __name__ == "__main__":
    main()
