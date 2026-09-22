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
import transformers.modeling_utils as _mu
_original_dispatch_model = _mu.dispatch_model
def _dispatch_model_noop(model, **kwargs):
    return model
_mu.dispatch_model = _dispatch_model_noop

import argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from bench_common import ArtifactSpec, run_all, screen_mse, filter_specs, SubmoduleTimer


CKPT = "/workspace/ckpts/openvla-libero-spatial"
AWQ_CKPT = "/workspace/ckpts/openvla-libero-spatial-awq"
UNNORM_KEY = "libero_spatial"   # 실제 checkpoint의 dataset_statistics.json 키와 일치 확인됨
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
PROMPT_TMPL = "In: What action should the robot take to {instr}?\nOut:"

# [신규] 공식 openvla_utils.py의 get_vla_action()이 center_crop=True일 때 적용하는
# 전처리. run_libero_eval.py의 GenerateConfig 기본값 자체가 center_crop=True라,
# 표준 배포 checkpoint는 이 crop을 전제로 평가된다. 원본은 TensorFlow로 구현돼
# 있으나(tf.image.crop_and_resize), Jetson 컨테이너에 TF를 새로 설치하는 위험을
# 피하기 위해 수학적으로 동일한 결과를 내는 PIL/numpy 버전으로 재구현했다.
CENTER_CROP = True
CROP_SCALE = 0.9   # 면적 기준 0.9배. 변 길이는 sqrt(0.9)배(공식 주석의 경고 그대로 반영)


def crop_and_resize_pil(image: Image.Image, crop_scale: float = CROP_SCALE,
                        out_size=(224, 224)) -> Image.Image:
    """
    TF 버전 crop_and_resize(공식 openvla_utils.py)와 수학적으로 동일한 결과를 내는
    PIL/numpy 구현. 원본 중앙에서 면적 crop_scale배(변 길이 sqrt(crop_scale)배)만큼
    잘라낸 뒤 out_size로 리사이즈한다. 학습 시 random-crop augmentation을 쓴
    checkpoint를 center_crop으로 평가할 때 분포 이동(distribution shift)을 줄인다.
    """
    side_scale = np.sqrt(crop_scale)
    w, h = image.size  # PIL: (width, height)
    new_w, new_h = w * side_scale, h * side_scale
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize(out_size, Image.BILINEAR)


# --- 공유 전처리 / 추론 -------------------------------------------------------
def preprocess_openvla(extras, obs_dict):
    """obs_dict = {"image": HxWx3 uint8, "instruction": str} → 모델 입력."""
    proc = extras["processor"]
    img = obs_dict["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    img = img.convert("RGB")

    # [신규] center crop — 공식 get_vla_action()과 동일한 위치(프롬프트 빌드 전,
    # processor 호출 전)에 적용. 표준 checkpoint가 이 전제로 평가되므로 필수.
    if CENTER_CROP:
        img = crop_and_resize_pil(img, CROP_SCALE)

    prompt = PROMPT_TMPL.format(instr=obs_dict.get("instruction", "pick up the object"))
    inputs = proc(prompt, img)
    input_dtype = extras.get("input_dtype", DTYPE)
    inputs = {k: (v.to(DEVICE, dtype=input_dtype) if torch.is_floating_point(v) else v.to(DEVICE))
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
    with torch.device("cuda"):
        model = AutoModelForVision2Seq.from_pretrained(
            CKPT, trust_remote_code=True, 
            quantization_config=bnb, device_map=None)
    model.eval()
    # 실제 vision backbone dtype을 찾아 extras에 기록
    actual_dtype = next(p.dtype for n, p in model.named_parameters() if "vision" in n or "featurizer" in n)
    return model, {"processor": _proc(), "input_dtype": actual_dtype}

def load_int8_no_outlier():
    """A1b: INT8이되 mixed-precision outlier 분해를 비활성화한 버전.

    llm_int8_threshold=0.0은 "모든 column을 int8로 양자화하고 분해를 하지 않는다"는
    뜻이다(공식 문서 기준). A1(기본 threshold=6.0, 분해 O)과 비교하면 '분해 알고리즘
    자체의 비용'이 분리되고, A2(NF4)와 비교하면 '비트 수 효과'가 분리된다.

    주의: 분해를 끄면 활성화 이상치가 int8로 뭉개져 정확도가 떨어질 수 있다
    (이게 원래 분해가 존재하는 이유다). 따라서 이 artifact는 '속도 원인 규명용
    ablation'이지, 실제 배포 후보가 아니다 — MSE 스크리닝으로 정확도 손실을 반드시 확인한다.
    """
    bnb = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=0.0,   # ← 0.0이 "분해 비활성화". inf 아님.
    )
    with torch.device("cuda"):
        model = AutoModelForVision2Seq.from_pretrained(
            CKPT, trust_remote_code=True, quantization_config=bnb, device_map=None,
        )
    model.eval()
    actual_dtype = next(p.dtype for n, p in model.named_parameters()
                        if "vision" in n or "featurizer" in n)
    return model, {"processor": _proc(), "input_dtype": actual_dtype}

def load_int4_bnb():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # ← 명시적으로 지정 (기본값 fp32 방지)
        bnb_4bit_use_double_quant=True,
    )
    with torch.device("cuda"):
        model = AutoModelForVision2Seq.from_pretrained(
            CKPT, trust_remote_code=True, quantization_config=bnb, device_map=None,
        )
    model.eval()
    actual_dtype = next(p.dtype for n, p in model.named_parameters() if "vision" in n or "featurizer" in n)
    return model, {"processor": _proc(), "input_dtype": actual_dtype}


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
    "A0_fp16": load_fp16, "A1_int8": load_int8_bnb,
    "A1b_int8_nooutlier":load_int8_no_outlier, "A2_int4": load_int4_bnb,
    "A3_int4_awq": load_awq, "A4_int4_compile": load_int4_compiled,
    "A5_token_prune": load_token_prune,
}build_inputs=build_inputs_openvla,
                 notes="nf4 + compile", **COMMON),
    ArtifactSpec(name="


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
    ArtifactSpec(name="A1b_int8_nooutlier", precision="int8", technique="no_outlier",
                 load_model=load_int8_no_outlier, build_inputs=build_inputs_openvla,
                 notes="bnb 8bit, llm_int8_threshold=0.0 (분해 비활성화, ablation용)", **COMMON), 
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
