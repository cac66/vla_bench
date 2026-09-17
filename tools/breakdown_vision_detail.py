"""
tools/breakdown_vision_detail.py  (AGX Orin 컨테이너 안에서 실행)
------------------------------------------------------------------
왜 이 스크립트가 필요한가
  torch.profiler(CUPTI 기반) 커널 단위 프로파일링이 이 Jetson에서 막혀 있다
  (CUPTI가 x86 NVIDIA 드라이버 전용 구조라 L4T/Tegra에는 그 경로 자체가 없음,
  /proc/driver/nvidia/capabilities/profiler-device 자체가 존재하지 않음을 확인함).
  CUPTI 없이도 "vision_backbone 안 어디가 느린가"를 알아내기 위해, CUPTI가 아니라
  torch.cuda.Event(CUDA 런타임 API, CUPTI와 무관하게 이미 정상 동작 확인됨)로
  vision_backbone의 하위 구조를 직접 쪼개서 잰다.

무엇을 재는가 (breakdown_compare.py보다 한 단계 더 세분화)
  vision_backbone은 두 개의 ViT(VisionTransformer)로 구성돼 있다:
    featurizer, fused_featurizer  (OpenVLA의 dual vision encoder 구조)
  각각의 내부를 다시 4개 그룹으로 나눈다:
    patch_embed       : 이미지를 patch로 자르는 첫 conv 레이어
    attn(all blocks)  : 모든 transformer block의 attention 부분 합
    mlp(all blocks)   : 모든 transformer block의 MLP(feed-forward) 부분 합
    norm(all blocks)  : 모든 transformer block의 LayerNorm 부분 합

  bnb 4bit/8bit 양자화 레이어는 attn(qkv projection)과 mlp(fc1/fc2)의 Linear에
  걸려있다(verify_quant.py에서 확인된 샘플이 attn.qkv였다). 따라서 attn/mlp 그룹의
  배수가 patch_embed/norm보다 훨씬 크게 나오면, "양자화된 Linear 레이어 자체의
  dequantize 비용"이 원인이라는 게 구조적으로 확정된다.

실행 예)
  python3 tools/breakdown_vision_detail.py --artifacts A0_fp16 A1_int8 A2_int4
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_openvla as M
from bench_common import free_model


class GroupedTimer:
    """
    SubmoduleTimer(bench_common.py)의 확장판: 여러 모듈을 하나의 라벨로 묶어
    합산할 수 있다(예: 12개 block의 attn을 전부 "attn(all blocks)"로 합산).
    CUDA Event 기반이라 CUPTI 권한 문제와 무관하게 동작한다.
    """
    def __init__(self, groups: dict):
        self.groups = groups  # {label: [module, module, ...]}
        self._handles = []
        self._acc = {label: 0.0 for label in groups}

    def __enter__(self):
        for label, modules in self.groups.items():
            for mod in modules:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)

                def pre(m, i, _s=s):
                    _s.record()

                def post(m, i, o, _s=s, _e=e, _label=label):
                    _e.record()
                    torch.cuda.synchronize()
                    self._acc[_label] += _s.elapsed_time(_e)

                self._handles.append(mod.register_forward_pre_hook(pre))
                self._handles.append(mod.register_forward_hook(post))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()

    def result_ms(self, n_iters: int) -> dict:
        return {label: v / max(n_iters, 1) for label, v in self._acc.items()}


def build_vit_groups(vit_module, prefix: str) -> dict:
    """하나의 VisionTransformer(featurizer 또는 fused_featurizer)에서
    patch_embed/attn/mlp/norm 그룹을 만든다. timm ViT의 표준 구조를 가정한다."""
    groups = {}

    if hasattr(vit_module, "patch_embed"):
        groups[f"{prefix}.patch_embed"] = [vit_module.patch_embed]

    blocks = list(getattr(vit_module, "blocks", []))
    attn_mods = [b.attn for b in blocks if hasattr(b, "attn")]
    mlp_mods = [b.mlp for b in blocks if hasattr(b, "mlp")]
    norm_mods = []
    for b in blocks:
        if hasattr(b, "norm1"):
            norm_mods.append(b.norm1)
        if hasattr(b, "norm2"):
            norm_mods.append(b.norm2)

    if attn_mods:
        groups[f"{prefix}.attn(all {len(attn_mods)} blocks)"] = attn_mods
    if mlp_mods:
        groups[f"{prefix}.mlp(all {len(mlp_mods)} blocks)"] = mlp_mods
    if norm_mods:
        groups[f"{prefix}.norm(all {len(norm_mods)} blocks)"] = norm_mods

    return groups


def run_one(artifact_name: str, warmup: int = 5, n: int = 20) -> dict:
    print(f"\n{'='*60}\n[vision detail] {artifact_name}\n{'='*60}")

    if artifact_name not in M.LOADERS:
        print(f"  [error] LOADERS에 없음: {artifact_name}")
        return {"artifact": artifact_name, "error": "not_found"}

    model, extras = M.LOADERS[artifact_name]()
    inputs, seq_len, infer_fn = M.build_inputs_openvla(model, extras)

    vb = model.vision_backbone
    groups = {}
    # 최상위: featurizer 전체 vs fused_featurizer 전체 (1단계 비교, 참고용)
    if hasattr(vb, "featurizer"):
        groups["featurizer(전체)"] = [vb.featurizer]
        groups.update(build_vit_groups(vb.featurizer, "featurizer"))
    if hasattr(vb, "fused_featurizer"):
        groups["fused_featurizer(전체)"] = [vb.fused_featurizer]
        groups.update(build_vit_groups(vb.fused_featurizer, "fused_featurizer"))

    if not groups:
        print("  [error] featurizer/fused_featurizer를 못 찾음. model.vision_backbone 구조 확인 필요.")
        free_model(model)
        return {"artifact": artifact_name, "error": "no_groups"}

    print(f"  워밍업 {warmup}회 …")
    for _ in range(warmup):
        _ = infer_fn(model, inputs, extras)
    torch.cuda.synchronize()

    print(f"  세부 breakdown 측정 중 ({n}회) …")
    with GroupedTimer(groups) as t:
        for _ in range(n):
            _ = infer_fn(model, inputs, extras)

    res = t.result_ms(n)
    for label, ms in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"    {ms:>9.3f} ms  {label}")

    free_model(model)
    return {"artifact": artifact_name, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", nargs="+", default=["A0_fp16", "A1_int8", "A2_int4"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    results = [run_one(a, warmup=args.warmup, n=args.n) for a in args.artifacts]
    valid = [r for r in results if "error" not in r]
    if len(valid) < 2:
        print("\n비교할 artifact가 2개 미만이라 비교표를 생략한다.")
        return

    base = valid[0]
    # base(A0)에 있는 라벨 기준으로, 다른 artifact와의 배수를 계산
    labels = [k for k in base if k != "artifact"]

    print(f"\n{'='*70}\n[비교표] {base['artifact']} 대비 배수 (내림차순 정렬은 A1 기준)\n{'='*70}")
    header = f"{'component':<38}" + "".join(f"{r['artifact']:>16}" for r in valid)
    print(header)
    print("-" * len(header))

    # A1(두번째 artifact가 있으면)의 배수 기준으로 정렬해, 가장 크게 벌어진 component를 위로
    sort_key_artifact = valid[1]["artifact"] if len(valid) > 1 else valid[0]["artifact"]

    def ratio_for(label, r):
        b = base.get(label, 0.0)
        v = r.get(label, 0.0)
        if b <= 1e-6:
            return float("nan")
        return v / b

    # 두 번째 artifact(보통 A1) 기준 배수가 큰 순서로 정렬해, 가장 크게 벌어진
    # component가 표 맨 위로 오게 한다. NaN(base가 거의 0)은 맨 뒤로 보낸다.
    if len(valid) > 1:
        sort_target = valid[1]

        def sort_key(lb):
            ratio = ratio_for(lb, sort_target)
            return ratio if ratio == ratio else -1.0  # ratio==ratio가 False면 NaN
        labels_sorted = sorted(labels, key=sort_key, reverse=True)
    else:
        labels_sorted = labels

    for lb in labels_sorted:
        row = f"{lb:<38}"
        for r in valid:
            ratio = ratio_for(lb, r)
            row += f"{ratio:>14.2f}x" if ratio == ratio else f"{'N/A':>15}"
        print(row)

    print("\n[해석 가이드]")
    print("  - attn/mlp 그룹의 배수가 patch_embed/norm 그룹보다 뚜렷이 크면,")
    print("    양자화된 Linear 레이어(qkv projection, fc1/fc2)의 dequantize 비용이")
    print("    vision 쪽 병목의 직접 원인이라는 게 구조적으로 확정된다.")
    print("  - featurizer와 fused_featurizer 중 한쪽만 유독 배수가 크면,")
    print("    두 ViT의 입력 해상도/패치 수 차이가 원인일 수 있다(둘 중 하나가")
    print("    더 많은 patch를 처리해 양자화 오버헤드가 선형으로 더 크게 곱해짐).")


if __name__ == "__main__":
    main()
