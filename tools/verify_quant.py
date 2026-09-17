"""
tools/verify_quant.py  (AGX Orin 컨테이너 안에서 실행)
------------------------------------------------------
"import가 성공했다"와 "실제로 4bit/8bit로 양자화돼 로드됐다"는 다른 것이다.
이 스크립트는 지정한 artifact를 로드해 다음을 교차 확인한다.

  1) 모델의 Linear 레이어가 실제로 bnb.nn.Linear4bit/Linear8bitLt 클래스인가
     (nn.Linear 그대로면 양자화가 전혀 안 걸린 것)
  2) 그 레이어의 quant_state가 기대한 값(quant_type=nf4, compute_dtype=float16)과 일치하는가
  3) 실제 GPU 메모리 사용량이 이론적 기대 범위(FP16 대비 축소)에 들어오는가
  4) (여러 artifact를 인자로 주면) 서로 비교해 표로 보여준다

실행 예)
  python3 tools/verify_quant.py --artifacts A0_fp16 A1_int8 A2_int4
"""

import argparse
import sys

import torch
import os

# tools/ 안에서 실행해도 상위 폴더(vla_bench/)의 bench_openvla.py를 찾도록 경로 추가.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# bitsandbytes의 실제 Linear 클래스. 없으면 quant 레이어 판별이 불가하므로 명확히 알림.
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def count_layer_types(model):
    """모델 내 Linear 계열 레이어를 클래스별로 센다."""
    counts = {"nn.Linear(양자화 안 됨)": 0}
    if HAS_BNB:
        counts["bnb.Linear4bit"] = 0
        counts["bnb.Linear8bitLt"] = 0

    sample_4bit_module = None
    sample_8bit_module = None

    for name, module in model.named_modules():
        if HAS_BNB and isinstance(module, bnb.nn.Linear4bit):
            counts["bnb.Linear4bit"] += 1
            if sample_4bit_module is None:
                sample_4bit_module = (name, module)
        elif HAS_BNB and isinstance(module, bnb.nn.Linear8bitLt):
            counts["bnb.Linear8bitLt"] += 1
            if sample_8bit_module is None:
                sample_8bit_module = (name, module)
        elif isinstance(module, torch.nn.Linear):
            counts["nn.Linear(양자화 안 됨)"] += 1

    return counts, sample_4bit_module, sample_8bit_module


def inspect_quant_state(sample_module, kind: str):
    """샘플 레이어 하나의 quant_state(양자화 세부설정)를 읽어 출력용 dict로 만든다."""
    if sample_module is None:
        return None
    name, module = sample_module
    info = {"layer_name": name, "weight_dtype": str(module.weight.dtype)}

    if kind == "4bit":
        qs = getattr(module.weight, "quant_state", None)
        if qs is not None:
            info["quant_type"] = getattr(qs, "quant_type", "알수없음")
            info["compute_dtype"] = str(getattr(qs, "dtype", "알수없음"))
            info["blocksize"] = getattr(qs, "blocksize", "알수없음")
        else:
            info["quant_state"] = "없음(양자화 안 된 것으로 보임)"
    elif kind == "8bit":
        info["has_fp16_weights"] = getattr(module, "has_fp16_weights", "알수없음")
        info["SCB_존재"] = getattr(module.weight, "SCB", None) is not None

    return info


def gpu_memory_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


def verify_one(artifact_name: str, loaders: dict) -> dict:
    print(f"\n{'='*60}\n[검증] {artifact_name}\n{'='*60}")

    if artifact_name not in loaders:
        print(f"  [error] LOADERS에 '{artifact_name}' 없음. 가능: {list(loaders)}")
        return {"artifact": artifact_name, "error": "not_found"}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_memory_mb()

    model, extras = loaders[artifact_name]()

    mem_after = gpu_memory_mb()
    mem_used = mem_after - mem_before

    counts, sample_4bit, sample_8bit = count_layer_types(model)

    print(f"  GPU 메모리 사용량: {mem_used:.1f} MB")
    print(f"  레이어 타입 분포: {counts}")

    detail = None
    if counts.get("bnb.Linear4bit", 0) > 0:
        detail = inspect_quant_state(sample_4bit, "4bit")
        print(f"  4bit 샘플 레이어 상세: {detail}")
    elif counts.get("bnb.Linear8bitLt", 0) > 0:
        detail = inspect_quant_state(sample_8bit, "8bit")
        print(f"  8bit 샘플 레이어 상세: {detail}")
    else:
        print("  [주의] 양자화 레이어(Linear4bit/Linear8bitLt)가 하나도 없다 — "
              "이 artifact가 실제로는 양자화되지 않았을 수 있다.")

    # 모델은 확인 후 반드시 해제 (다음 artifact 비교를 위해 메모리 오염 방지)
    del model
    torch.cuda.empty_cache()

    return {
        "artifact": artifact_name,
        "gpu_mem_mb": round(mem_used, 1),
        "layer_counts": counts,
        "sample_detail": detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", nargs="+", default=["A0_fp16", "A1_int8", "A2_int4"])
    ap.add_argument("--model", choices=["openvla", "smolvla"], default="openvla")
    args = ap.parse_args()

    if not HAS_BNB:
        print("[warn] bitsandbytes를 import할 수 없다 — Linear4bit/8bit 클래스 판별이 불가능하다. "
              "메모리량만으로 대략 추정하지만, 정확한 검증은 bitsandbytes가 있어야 한다.")

    if args.model == "openvla":
        import bench_openvla as M
    else:
        import bench_smolvla as M

    results = [verify_one(name, M.LOADERS) for name in args.artifacts]

    # --- 요약표 ---
    print(f"\n{'='*60}\n[요약]\n{'='*60}")
    print(f"{'artifact':<16} {'GPU_MB':>10}  {'4bit레이어':>10}  {'8bit레이어':>10}  {'미양자화Linear':>14}")
    for r in results:
        if "error" in r:
            print(f"{r['artifact']:<16}  (로드 실패)")
            continue
        c = r["layer_counts"]
        print(f"{r['artifact']:<16} {r['gpu_mem_mb']:>10.1f}  "
              f"{c.get('bnb.Linear4bit', 0):>10}  {c.get('bnb.Linear8bitLt', 0):>10}  "
              f"{c.get('nn.Linear(양자화 안 됨)', 0):>14}")

    print("\n[판정 기준]")
    print("  - A2(int4)는 'bnb.Linear4bit' 개수가 0보다 커야 하고, quant_type='nf4' 확인 필요.")
    print("  - A2의 GPU_MB가 A0(fp16)보다 뚜렷이 작아야 한다(대략 1/3~1/4 수준 기대).")
    print("  - compute_dtype이 'torch.float32'로 나오면 아직 fp32 낙하 문제가 남은 것이다.")


if __name__ == "__main__":
    main()
