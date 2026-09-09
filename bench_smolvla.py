"""
bench_smolvla.py
----------------
Orin Nano 8GB + SmolVLA (LeRobot).

artifact ↔ 메타데이터
  B0 fp16/pytorch/none | B1 fp16/pytorch/step_reduce(스윕) | B2 int8/pytorch/none | B3 fp16/pytorch/torch_compile

주의(버전 의존)
- flow-matching step 수 attribute(예: policy.config.num_steps)는 설치된 LeRobot에서 확인.
- chunk 재생성 비용을 재려면 매 호출 policy.reset()으로 큐를 비운다.
"""

import argparse
import numpy as np
import torch

from bench_common import ArtifactSpec, run_all, screen_mse, filter_specs
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "HuggingFaceVLA/smolvla_libero"
DEVICE = "cuda:0"
CHUNK = 50
DEFAULT_FLOW_STEPS = 10


# --- 공유 전처리 / 추론 -------------------------------------------------------
def _to_chw(img):
    """HxWx3 uint8 → 1x3xHxW float[0,1] on device."""
    t = torch.as_tensor(np.asarray(img), dtype=torch.float32, device=DEVICE) / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def preprocess_smolvla(extras, obs_dict):
    """obs_dict = {"image","image2","state","instruction"} → LeRobot batch."""
    batch = {
        "observation.images.image":  _to_chw(obs_dict["image"]),
        "observation.images.image2": _to_chw(obs_dict.get("image2", obs_dict["image"])),
        "observation.state": torch.as_tensor(
            np.asarray(obs_dict.get("state", np.zeros(8, np.float32)), np.float32),
            device=DEVICE).unsqueeze(0),
        "task": [obs_dict.get("instruction", "pick up the object")],
    }
    pre = extras.get("preprocess")
    return pre(batch) if pre else batch


@torch.no_grad()
def predict_smolvla(policy, extras, obs_dict) -> np.ndarray:
    """실제 관측 → action(np). MSE·serving 공용. chunk의 첫 action 반환."""
    batch = preprocess_smolvla(extras, obs_dict)
    policy.reset()
    a = policy.select_action(batch)
    return np.asarray(a.detach().float().cpu().reshape(-1), dtype=np.float32)


def _dummy_obs():
    return {"image": np.random.randint(0, 255, (256, 256, 3), np.uint8),
            "image2": np.random.randint(0, 255, (256, 256, 3), np.uint8),
            "state": np.zeros(8, np.float32), "instruction": "pick up the black bowl"}


def build_inputs_smolvla(policy, extras):
    batch = preprocess_smolvla(extras, _dummy_obs())
    seq_len = 64  # 텍스트 토큰 대략치

    def infer(p, ins, ex):
        p.reset()                       # chunk 강제 재생성 → 실제 추론 비용 측정
        with torch.no_grad():
            return p.select_action(ins)

    return batch, seq_len, infer


# --- 로더들 -------------------------------------------------------------------
def _make_extras(policy):
    extras = {}
    try:  # LeRobot 전/후처리기(버전에 따라 경로 다름)
        from lerobot.policies.factory import make_pre_post_processors
        pre, post = make_pre_post_processors(
            policy.config, CKPT,
            preprocessor_overrides={"device_processor": {"device": DEVICE}})
        extras["preprocess"], extras["postprocess"] = pre, post
    except Exception as e:
        print(f"[warn] pre/post processor 미사용({e}) — raw batch로 진행")
    return extras


def load_bf16():
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEVICE).eval()
    return policy, _make_extras(policy)


def load_step_reduced(n_steps):
    policy, extras = load_bf16()
    try:
        policy.config.num_steps = n_steps   # TODO: 실제 attribute 확인
    except Exception as e:
        print(f"[warn] step 수 설정 실패: {e}")
    extras["flow_steps"] = n_steps
    return policy, extras


def load_int8():
    policy, extras = load_bf16()
    try:
        torch.quantization.quantize_dynamic(policy, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    except Exception as e:
        print(f"[warn] dynamic quant 미적용(대안 bnb 8bit): {e}")
    return policy, extras


def load_compiled():
    policy, extras = load_bf16()
    try:
        policy.select_action = torch.compile(policy.select_action, mode="reduce-overhead")
    except Exception as e:
        print(f"[warn] torch.compile 미적용: {e}")
    return policy, extras


LOADERS = {
    "B0_fp16": load_bf16, "B2_int8": load_int8, "B3_compile": load_compiled,
    **{f"B1_step{n}": (lambda n=n: load_step_reduced(n)) for n in (5, 3, 2)},
}


# --- registry -----------------------------------------------------------------
COMMON = dict(device="orin_nano_8gb", model="smolvla", runtime="pytorch",
              action_chunk_size=CHUNK, replan_interval="NA",
              warmup_iters=30, measure_iters=200, interval_ms=100, out_root="./benchmark")

ARTIFACTS = [
    ArtifactSpec(name="B0_fp16", precision="fp16", technique="none",
                 load_model=load_bf16, build_inputs=build_inputs_smolvla,
                 notes=f"baseline flow_steps={DEFAULT_FLOW_STEPS}", **COMMON),
    *[ArtifactSpec(name=f"B1_step{n}", precision="fp16", technique="step_reduce",
                   load_model=(lambda n=n: load_step_reduced(n)),
                   build_inputs=build_inputs_smolvla, notes=f"flow_steps={n}", **COMMON)
      for n in (5, 3, 2)],
    ArtifactSpec(name="B2_int8", precision="int8", technique="none",
                 load_model=load_int8, build_inputs=build_inputs_smolvla,
                 notes="속도/전력 목적", **COMMON),
    ArtifactSpec(name="B3_compile", precision="fp16", technique="torch_compile",
                 load_model=load_compiled, build_inputs=build_inputs_smolvla,
                 notes="compile", **COMMON),
]


# --- MSE용 관측 로더 (LeRobot dataset에서) ------------------------------------
def load_eval_obs_smolvla(repo_id="HuggingFaceVLA/libero", n=200):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id)
    obs_list, gts = [], []
    for i in range(min(n, len(ds))):
        f = ds[i]
        obs_list.append({
            "image": (f["observation.images.image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            "image2": (f["observation.images.image2"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                       if "observation.images.image2" in f else None,
            "state": f.get("observation.state", torch.zeros(8)).numpy(),
            "instruction": f.get("task", "pick up the object"),
        })
        if "action" in f:
            gts.append(f["action"].numpy().reshape(-1))
    gt = np.stack(gts) if gts else None
    return obs_list, gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["latency", "mse"], default="latency")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--repo", default="HuggingFaceVLA/libero")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    specs = filter_specs(ARTIFACTS, args.only)

    if args.mode == "latency":
        run_all(specs)
    else:
        obs_list, gt = load_eval_obs_smolvla(args.repo, args.n)
        screen_mse(specs, obs_list, predict_smolvla,
                   out_csv="./benchmark/mse_screen_smolvla.csv",
                   reference_name="B0_fp16", gt_actions=gt)
