"""
tools/breakdown_compare.py  (AGX Orin 컨테이너 안에서 실행)
------------------------------------------------------------
왜 이 스크립트가 필요한가
  A1(bnb 8bit)이 A0(fp16)보다 3.3배, A2(bnb 4bit)보다도 2배 느리다는 게 실측으로
  확인됐다(latency·energy 교차검증 완료). 다음 질문은 "그 시간이 모델의 어느
  부분(vision backbone vs language model)에서 새는가"다.

  bench_openvla.py의 breakdown_openvla()가 이미 이 기능(SubmoduleTimer로 vision/
  backbone 각각의 누적 forward 시간을 재는 것)을 갖고 있다 — 이 스크립트는 그걸
  재사용해서, A0/A1/A2를 "같은 조건으로 연속 실행 → 결과를 한 표로 비교"하는
  용도로 감싼 것이다. 새 측정 로직을 만들지 않고 기존 코드를 재사용한다.

무엇을 보게 되는가
  - vision_ms  : vision_backbone(이미지 인코더) 누적 forward 시간(1회 predict_action당 평균)
  - backbone_ms: language_model(LLM, autoregressive decode 포함) 누적 forward 시간
  - 이 둘의 합 대비 실제 predict_action() 전체 wall-clock 시간과의 비율도 함께 낸다
    (hook이 못 잡는 오버헤드, 즉 "vision도 backbone도 아닌 어딘가"가 있는지 보기 위해)

판단 기준
  - A0 대비 A1에서 vision_ms 비율이 비슷한데 backbone_ms만 튀면
    → 8bit outlier 분해가 language_model(주로 Linear 레이어가 몰린 곳)에서 일어난다는
      가설과 일치. 다음 단계(torch.profiler)에서 그 안의 어느 커널인지 파고든다.
  - vision_ms까지 같이 튀면 → vision_backbone에도 Linear4bit/8bit 레이어가 있다는 뜻이니
    (실제로 verify_quant.py 결과에서 vision_backbone.featurizer... 레이어가 양자화 대상에
    있었다) 범위를 vision까지 넓혀 다시 봐야 한다.

실행 예)
  python3 tools/breakdown_compare.py --artifacts A0_fp16 A1_int8 A2_int4
"""

import argparse
import os
import sys
import time

import torch

# tools/ 안에서 실행해도 상위 폴더(vla_bench/)의 bench_openvla.py를 찾도록 경로 추가.
# python3 tools/breakdown_compare.py 로 실행하면 sys.path엔 tools/만 들어가고
# 부모 폴더는 안 들어가서 import가 실패하는 문제를 해결한다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_openvla as M
from bench_common import free_model


def timed_predict_action(model, inputs, n=20):
    """breakdown과 별개로, predict_action() 전체의 순수 wall-clock 시간도 함께 잰다.
    hook이 vision+backbone으로 못 잡는 부분(예: de-tokenization, un-normalize)이
    있는지 대조하기 위한 기준선이다."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _ = model.predict_action(**inputs, unnorm_key=M.UNNORM_KEY, do_sample=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0  # ms/call


def run_one(artifact_name: str, warmup: int = 5, n_breakdown: int = 20) -> dict:
    print(f"\n{'='*60}\n[breakdown] {artifact_name}\n{'='*60}")

    if artifact_name not in M.LOADERS:
        print(f"  [error] LOADERS에 없음: {artifact_name}")
        return {"artifact": artifact_name, "error": "not_found"}

    model, extras = M.LOADERS[artifact_name]()
    inputs, seq_len, infer_fn = M.build_inputs_openvla(model, extras)

    # 워밍업: 첫 호출은 CUDA 커널 컴파일/캐시 때문에 비정상적으로 느릴 수 있어 제외한다.
    print(f"  워밍업 {warmup}회 …")
    for _ in range(warmup):
        _ = infer_fn(model, inputs, extras)
    torch.cuda.synchronize()

    # 1) component breakdown (기존 breakdown_openvla 재사용)
    print(f"  component breakdown 측정 중 ({n_breakdown}회) …")
    bd = M.breakdown_openvla(model, inputs, extras)

    # 2) 전체 wall-clock 기준선 (breakdown이 못 잡는 부분이 있는지 대조용)
    total_ms = timed_predict_action(model, inputs, n=n_breakdown)

    vision_ms = bd.get("vision_ms", 0.0)
    backbone_ms = bd.get("backbone_ms", 0.0)
    hooked_sum = vision_ms + backbone_ms
    unaccounted_ms = max(total_ms - hooked_sum, 0.0)

    result = {
        "artifact": artifact_name,
        "vision_ms": round(vision_ms, 3),
        "backbone_ms": round(backbone_ms, 3),
        "hooked_sum_ms": round(hooked_sum, 3),
        "total_ms": round(total_ms, 3),
        "unaccounted_ms": round(unaccounted_ms, 3),
        "vision_pct": round(100 * vision_ms / total_ms, 1) if total_ms else 0.0,
        "backbone_pct": round(100 * backbone_ms / total_ms, 1) if total_ms else 0.0,
        "unaccounted_pct": round(100 * unaccounted_ms / total_ms, 1) if total_ms else 0.0,
    }
    print(f"  vision={vision_ms:.1f}ms  backbone={backbone_ms:.1f}ms  "
          f"전체={total_ms:.1f}ms  미포착={unaccounted_ms:.1f}ms")

    free_model(model)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", nargs="+", default=["A0_fp16", "A1_int8", "A2_int4"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n", type=int, default=20, help="breakdown/전체시간 측정 반복 횟수")
    args = ap.parse_args()

    results = [run_one(a, warmup=args.warmup, n_breakdown=args.n) for a in args.artifacts]

    # --- 비교표 ---
    print(f"\n{'='*70}\n[비교표] component별 절대시간(ms) 및 비중(%)\n{'='*70}")
    header = f"{'artifact':<14}{'vision_ms':>10}{'backbone_ms':>13}{'total_ms':>10}{'미포착_ms':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['artifact']:<14}  (로드 실패)")
            continue
        print(f"{r['artifact']:<14}{r['vision_ms']:>10.1f}{r['backbone_ms']:>13.1f}"
              f"{r['total_ms']:>10.1f}{r['unaccounted_ms']:>10.1f}")

    print(f"\n{'='*70}\n[비교표] baseline(첫 artifact) 대비 배수\n{'='*70}")
    valid = [r for r in results if "error" not in r]
    if valid:
        base = valid[0]
        print(f"기준: {base['artifact']}")
        header2 = f"{'artifact':<14}{'vision_배수':>12}{'backbone_배수':>14}{'total_배수':>11}"
        print(header2)
        print("-" * len(header2))
        for r in valid:
            v_ratio = r["vision_ms"] / base["vision_ms"] if base["vision_ms"] else float("nan")
            b_ratio = r["backbone_ms"] / base["backbone_ms"] if base["backbone_ms"] else float("nan")
            t_ratio = r["total_ms"] / base["total_ms"] if base["total_ms"] else float("nan")
            print(f"{r['artifact']:<14}{v_ratio:>11.2f}x{b_ratio:>13.2f}x{t_ratio:>10.2f}x")

    print("\n[해석 가이드]")
    print("  - backbone_배수가 vision_배수보다 훨씬 크면 → language_model(LLM) 쪽이 병목.")
    print("    다음 단계(torch.profiler)에서 language_model의 Linear 레이어 커널을 집중적으로 본다.")
    print("  - 미포착_ms가 크면 → predict_action() 안에 vision/backbone 호출로 안 잡히는")
    print("    구간(예: de-tokenization, unnorm_key 처리, generate() 루프 오버헤드)이 크다는 뜻.")
    print("    이 경우 breakdown 대상(hook 대상 모듈)을 더 세분화해야 한다.")


if __name__ == "__main__":
    main()

