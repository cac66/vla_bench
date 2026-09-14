"""
bench_openvla.py
----------------
AGX Orin 64GB + OpenVLA-7B artifact.

한 모델 지식(로딩·전처리·추론)을 세 용도가 공유한다.
- 트랙 1 latency : build_inputs_openvla (더미 관측)
- 스크리닝 MSE   : predict_openvla (실제 관측)
- serving        : predict_openvla (네트워크로 받은 관측)

artifact ↔ 메타데이터
  A0 fp16/pytorch/none  | A1 int8/pytorch/none(bnb 8bit) | A2 int4/pytorch/none(bnb nf4)
  A3 int4/pytorch/awq   | A4 int4/pytorch/torch_compile  | A5 fp16/pytorch/token_prune
"""

import argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from bench_common import ArtifactSpec, run_all, screen_mse, filter_specs, SubmoduleTimer

CKPT = "openvla/openvla-7b-finetuned-libero-spatial"   # suite에 맞게 교체
AWQ_CKPT = "./ckpts/openvla-libero-spatial-awq"
UNNORM_KEY = "libero_spatial_no_noops"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
PROMPT_TMPL = "In: What action should the robot take to {instr}?\nOut:"


# --- 공유 전처리 / 추론 -------------------------------------------------------
def preprocess_openvla(extras, obs_dict):
    """obs_dict = {"image": HxWx3 uint8, "instruction": str} → 모델 입력."""
    proc = extras["processor"]
    img = obs_dict["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    prompt = PROMPT_TMPL.format(instr=obs_dict.get("instruction", "pick up the object"))
    inputs = proc(prompt, img)
    inputs = {k: (v.to(DEVICE, dtype=DTYPE) if torch.is_floating_point(v) else v.to(DEVICE))
              for k, v in inputs.items()}
    return inputs


@torch.no_grad()
def predict_openvla(model, extras, obs_dict) -> np.ndarray:
    """실제 관측 → 7-DoF action(np). MSE·serving 공용."""
    inputs = preprocess_openvla(extras, obs_dict)
    action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    return np.asarray(action, dtype=np.float32)


def _dummy_obs():
    return {"image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "instruction": "pick up the black bowl"}


def build_inputs_openvla(model, extras):
    """트랙 1: 더미 관측으로 입력 고정 후 그 입력을 반복 추론."""
    inputs = preprocess_openvla(extras, _dummy_obs())
    seq_len = int(inputs["input_ids"].shape[-1])

    def infer(m, ins, ex):
        return m.predict_action(**ins, unnorm_key=UNNORM_KEY, do_sample=False)

    return inputs, seq_len, infer


# --- 로더들 -------------------------------------------------------------------
def _proc():
    return AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)


def load_fp16():
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=DTYPE, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    return model, {"processor": _proc()}


def load_int8_bnb():
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=DTYPE,
        quantization_config=bnb, device_map={"": 0}, low_cpu_mem_usage=True).eval()
    return model, {"processor": _proc()}


def load_int4_bnb():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=DTYPE,
        quantization_config=bnb, device_map={"": 0}, low_cpu_mem_usage=True).eval()
    return model, {"processor": _proc()}


def load_awq():
    model = AutoModelForVision2Seq.from_pretrained(
        AWQ_CKPT, trust_remote_code=True, torch_dtype=DTYPE,
        device_map={"": 0}, low_cpu_mem_usage=True).eval()
    return model, {"processor": _proc()}


def load_int4_compiled():
    model, extras = load_int4_bnb()
    try:
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print(f"[warn] torch.compile 미적용: {e}")
    return model, extras


def load_token_prune():
    model, extras = load_fp16()
    # TODO: EfficientVLA류 training-free visual token 감축 모듈을 여기서 주입.
    extras["note"] = "token_prune 모듈 주입 필요"
    return model, extras


LOADERS = {
    "A0_fp16": load_fp16, "A1_int8": load_int8_bnb, "A2_int4": load_int4_bnb,
    "A3_int4_awq": load_awq, "A4_int4_compile": load_int4_compiled,
    "A5_token_prune": load_token_prune,
}


# --- breakdown ----------------------------------------------------------------
def breakdown_openvla(model, inputs, extras):
    mods = {}
    if hasattr(model, "vision_backbone"):
        mods["vision"] = model.vision_backbone
    if hasattr(model, "language_model"):
        mods["backbone"] = model.language_model
    if not mods:
        raise RuntimeError("submodule 이름 확인 필요 (print(model)로 조정)")
    n = 20
    with SubmoduleTimer(mods) as t:
        for _ in range(n):
            _ = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    res = t.result_ms(n)
    for k in ("vision_ms", "backbone_ms", "action_ms"):
        res.setdefault(k, 0.0)
    return res


# --- registry -----------------------------------------------------------------
COMMON = dict(device="agx_orin_64gb", model="openvla", runtime="pytorch",
              action_chunk_size=1, replan_interval="NA",
              warmup_iters=30, measure_iters=200, interval_ms=100, out_root="./benchmark")

ARTIFACTS = [
    ArtifactSpec(name="A0_fp16", precision="fp16", technique="none",
                 load_model=load_fp16, build_inputs=build_inputs_openvla, notes="baseline", **COMMON),
    ArtifactSpec(name="A1_int8", precision="int8", technique="none",
                 load_model=load_int8_bnb, build_inputs=build_inputs_openvla, notes="bnb 8bit", **COMMON),
    ArtifactSpec(name="A2_int4", precision="int4", technique="none",
                 load_model=load_int4_bnb, build_inputs=build_inputs_openvla, notes="bnb nf4", **COMMON),
    ArtifactSpec(name="A3_int4_awq", precision="int4", technique="awq",
                 load_model=load_awq, build_inputs=build_inputs_openvla, notes="prebuilt AWQ", **COMMON),
    ArtifactSpec(name="A4_int4_compile", precision="int4", technique="torch_compile",
                 load_model=load_int4_compiled, build_inputs=build_inputs_openvla,
                 notes="nf4 + compile", **COMMON),
    ArtifactSpec(name="A5_token_prune", precision="fp16", technique="token_prune",
                 load_model=load_token_prune, build_inputs=build_inputs_openvla,
                 notes="visual token 감축", **COMMON),
]


# --- MSE용 관측 로더 ----------------------------------------------------------
def load_eval_obs_openvla(npz_path: str, n: int = 200):
    """
    npz(images[N,H,W,3] uint8, [선택] instructions[N] str, [선택] actions[N,7]) → (obs_list, gt).
    LIBERO 데모/녹화 rollout에서 미리 만들어 둔다.
    """
    d = np.load(npz_path, allow_pickle=True)
    imgs = d["images"][:n]
    instrs = d["instructions"][:n] if "instructions" in d else ["pick up the object"] * len(imgs)
    obs_list = [{"image": imgs[i], "instruction": str(instrs[i])} for i in range(len(imgs))]
    gt = d["actions"][:n] if "actions" in d else None
    return obs_list, gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["latency", "mse"], default="latency")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--eval-npz", default="./data/openvla_eval.npz", help="MSE용 관측 npz")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    specs = filter_specs(ARTIFACTS, args.only)

    if args.mode == "latency":
        if args.breakdown:
            for s in specs:
                s.breakdown_fn = breakdown_openvla
        run_all(specs)
    else:
        obs_list, gt = load_eval_obs_openvla(args.eval_npz, args.n)
        screen_mse(specs, obs_list, predict_openvla,
                   out_csv="./benchmark/mse_screen_openvla.csv", reference_name="A0_fp16",
                   gt_actions=gt)