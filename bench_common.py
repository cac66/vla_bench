"""
bench_common.py
---------------
모든 artifact가 공유하는 측정 하네스다.

두 측정 트랙을 지원한다.
- 트랙 1 (latency/power/memory) : run_artifact()  — TegraProfiler로 Jetson 단독 측정, 시뮬 불필요.
- 스크리닝 (open-loop MSE)       : screen_mse()   — 관측 집합에 통과, 시뮬 불필요.
(트랙 2 정확도=success rate는 serving/ 쪽 closed-loop에서 처리한다.)

설계 원칙
- 측정 loop는 여기서 고정한다. artifact마다 바꾸지 않는다.
- artifact는 "로딩 함수 + 입력/추론 콜백 + profiler 메타데이터"만 제공한다(ArtifactSpec).
- 한 번에 하나의 모델만 GPU에 올린다(측정 후 free) → memory/latency 오염 방지.
"""

import gc
import os
import csv
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

import torch
from tegra_profiler import TegraProfiler

from energy import SysfsPowerSampler, measure_idle_baseline, energy_summary


# ---------------------------------------------------------------------------
# Artifact 스펙
# ---------------------------------------------------------------------------
@dataclass
class ArtifactSpec:
    # --- profiler 메타데이터 (CSV 컬럼) ---
    device: str                      # "orin_nano_8gb" / "agx_orin_64gb"
    model: str                       # "openvla" / "smolvla"
    precision: str                   # "fp16" / "int8" / "int4"
    runtime: str                     # "pytorch" / "tensorrt" / "llamacpp"
    technique: str = "none"
    replan_interval: Any = "NA"
    action_chunk_size: int = 1
    warmup_iters: int = 30
    measure_iters: int = 200
    interval_ms: int = 100
    notes: str = ""
    out_root: str = "./benchmark"
    measure_energy: bool = True          # sysfs 에너지 측정 on/off
    idle_baseline_s: float = 30.0        # idle 대기 시간(확정값)

    # --- 실행 콜백 ---
    # load_model() -> (model, extras: dict)
    load_model: Callable[[], tuple] = None
    # build_inputs(model, extras) -> (inputs, seq_len:int, infer_fn)
    #   infer_fn(model, inputs, extras) -> Any
    build_inputs: Callable[[Any, dict], tuple] = None
    # 선택: breakdown(model, inputs, extras) -> dict(vision_ms, backbone_ms, action_ms)
    breakdown_fn: Optional[Callable[[Any, Any, dict], dict]] = None
    # 선택: accuracy() -> dict(success_rate_pct, action_mse, eval_env)
    accuracy_fn: Optional[Callable[[], dict]] = None

    name: str = field(default="")


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def free_model(model):
    """모델을 GPU에서 내려 다음 artifact가 깨끗한 상태에서 측정되게 한다."""
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def append_csv_row(path: str, row: dict) -> None:
    """헤더는 최초 1회만 쓰고 이후 행을 append."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# 트랙 1: latency/power/memory 측정
# ---------------------------------------------------------------------------
def run_artifact(spec: ArtifactSpec) -> None:
    """
    측정 순서(에너지 반영):
      ① idle-baseline(BE) 측정 — 모델 로딩 전, idle_baseline_s초 sysfs 폴링
      ② 모델 로딩 + 입력 준비
      ③ warmup (결과 제외)
      ④ measure — TegraProfiler(latency) + SysfsPowerSampler(전력, 병행) 동시 측정
      ⑤ TE(active power) → net = TE-BE → energy_j_total/net/energy_mj_per_action 계산
      ⑥ 성능 CSV(TegraProfiler)와 별개로 에너지 CSV row를 append (같은 name/device/precision/technique로 join 가능)
    """
    label = spec.name or f"{spec.model}-{spec.precision}-{spec.technique}"

    # ① idle-baseline: 모델을 올리기 전에 측정해야 "이 artifact 로딩·추론이 추가로 쓴" 전력을 분리할 수 있다.
    # [정정] AGX Orin은 단일 VDD_IN 채널이 없어 compute(1+2)/total(1+2+3) 두 값을 dict로 받는다.
    idle_baseline = {"compute": 0.0, "total": 0.0}
    if spec.measure_energy:
        idle_baseline = measure_idle_baseline(duration_s=spec.idle_baseline_s)

    print(f"\n=== [{label}] 로딩 ===")
    model, extras = spec.load_model()

    # 입력·seq_len은 profiler 생성 전에 확정한다.
    inputs, seq_len, infer_fn = spec.build_inputs(model, extras)

    prof = TegraProfiler(
        device=spec.device, model=spec.model, precision=spec.precision,
        runtime=spec.runtime, technique=spec.technique,
        replan_interval=spec.replan_interval, action_chunk_size=spec.action_chunk_size,
        seq_len=seq_len, warmup_iters=spec.warmup_iters, measure_iters=spec.measure_iters,
        interval_ms=spec.interval_ms, notes=spec.notes, out_root=spec.out_root,
    )

    print(f"=== [{label}] 측정 (warmup={spec.warmup_iters}, measure={spec.measure_iters}) ===")
    energy_sampler = SysfsPowerSampler(interval_s=0.01) if spec.measure_energy else None

    with prof:
        for _ in range(spec.warmup_iters):
            _ = infer_fn(model, inputs, extras)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        lat_ms = []
        # ④ measure 구간: TegraProfiler(latency)와 sysfs 전력 폴링을 같은 구간에서 동시 진행.
        if energy_sampler is not None:
            energy_sampler.__enter__()
        try:
            for _ in range(spec.measure_iters):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = infer_fn(model, inputs, extras)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                lat_ms.append((time.perf_counter() - t0) * 1000)
        finally:
            if energy_sampler is not None:
                energy_sampler.__exit__()
        prof.record_latency(lat_ms)

        if spec.breakdown_fn is not None:
            try:
                prof.record_latency_breakdown(**spec.breakdown_fn(model, inputs, extras))
            except Exception as e:
                print(f"[warn] breakdown 실패({label}): {e}")

        if spec.accuracy_fn is not None:
            try:
                prof.record_accuracy(**spec.accuracy_fn())
            except Exception as e:
                print(f"[warn] accuracy 기록 실패({label}): {e}")

    # ⑤~⑥ 에너지 요약 계산 및 별도 CSV 기록 (성능 CSV는 TegraProfiler가 이미 저장)
    # [정정] compute(GPU+CPU)와 total(+시스템 5V) 두 지표를 한 행에 함께 기록한다.
    if energy_sampler is not None:
        summary = energy_summary(energy_sampler, idle_baseline, spec.measure_iters)
        row = {"name": spec.name, "device": spec.device, "model": spec.model,
               "precision": spec.precision, "technique": spec.technique,
               "measure_iters": spec.measure_iters, **summary}
        append_csv_row(f"{spec.out_root}/energy.csv", row)
        print(f"[energy] {label}: "
              f"compute net={summary['net_power_w_compute']}W/{summary['energy_mj_per_action_compute']}mJ  "
              f"total net={summary['net_power_w_total']}W/{summary['energy_mj_per_action_total']}mJ")

    free_model(model)
    print(f"=== [{label}] 완료 → {spec.out_root} ===")


def run_all(specs, only=None) -> None:
    for spec in specs:
        if only and not any(k in (spec.name or "") for k in only):
            continue
        try:
            run_artifact(spec)
        except Exception as e:
            print(f"[error] artifact 실패({spec.name}): {e}")


def filter_specs(specs, only=None):
    """
    --only 인자로 받은 키워드 목록에 맞춰 ArtifactSpec 목록을 걸러낸다.
    bench_openvla.py/bench_smolvla.py의 __main__이 공통으로 쓰던 인라인 필터링을
    여기로 옮겨 중복을 없앴다. only가 비어있으면(None/빈 리스트) 전체를 그대로 반환한다.
    """
    if not only:
        return list(specs)
    return [s for s in specs if any(k in (s.name or "") for k in only)]


# ---------------------------------------------------------------------------
# 스크리닝: open-loop MSE (시뮬 불필요)
# ---------------------------------------------------------------------------
def screen_mse(specs, obs_list, predict_fn,
               out_csv: str = "./benchmark/mse_screen.csv",
               reference_name: str = "A0_fp16",
               gt_actions=None) -> dict:
    """
    각 artifact를 동일 관측 집합에 통과시켜 action을 모으고 MSE를 계산한다.

    - obs_list      : 관측 dict 리스트(모델별 predict_fn이 해석). 실제 LIBERO 관측 권장.
    - predict_fn    : predict_fn(model, extras, obs_dict) -> np.ndarray(action)
    - reference_name: 이 artifact 예측을 기준(FP16)으로 self-reference MSE 계산.
    - gt_actions    : (선택) [N, action_dim] ground-truth. 있으면 vs-GT MSE도 계산.

    quantization 스크리닝은 vs-FP16이 vs-GT보다 유용하고 train/test 누출도 우회한다.
    """
    import numpy as np

    ordered = sorted(specs, key=lambda s: 0 if s.name == reference_name else 1)
    ref_preds = None
    results = {}

    for spec in ordered:
        print(f"\n[MSE] {spec.name} 로딩·예측 …")
        model, extras = spec.load_model()
        preds = np.stack([
            np.asarray(predict_fn(model, extras, obs), dtype=np.float32).reshape(-1)
            for obs in obs_list
        ])
        free_model(model)

        row = {
            "name": spec.name, "device": spec.device, "model": spec.model,
            "precision": spec.precision, "technique": spec.technique,
            "n_samples": len(obs_list), "mse_vs_fp16": "NA", "mse_vs_gt": "NA",
        }
        if spec.name == reference_name:
            ref_preds = preds
            row["mse_vs_fp16"] = 0.0
        elif ref_preds is not None:
            row["mse_vs_fp16"] = float(np.mean((preds - ref_preds) ** 2))
        if gt_actions is not None:
            gt = np.asarray(gt_actions, dtype=np.float32)
            m = min(len(gt), len(preds))
            row["mse_vs_gt"] = float(np.mean((preds[:m] - gt[:m]) ** 2))

        append_csv_row(out_csv, row)
        results[spec.name] = row
        print(f"[MSE] {spec.name}: vs_fp16={row['mse_vs_fp16']} vs_gt={row['mse_vs_gt']}")
    return results


# ---------------------------------------------------------------------------
# 선택: forward hook 기반 간이 breakdown (PyTorch 경로)
# ---------------------------------------------------------------------------
class SubmoduleTimer:
    def __init__(self, modules: dict):
        self.modules = modules
        self._handles = []
        self._acc = {k: 0.0 for k in modules}

    def __enter__(self):
        for key, mod in self.modules.items():
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)

            def pre(m, i, _s=s):
                _s.record()

            def post(m, i, o, _s=s, _e=e, _k=key):
                _e.record(); torch.cuda.synchronize()
                self._acc[_k] += _s.elapsed_time(_e)

            self._handles += [mod.register_forward_pre_hook(pre),
                              mod.register_forward_hook(post)]
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()

    def result_ms(self, n_iters: int) -> dict:
        return {f"{k}_ms": v / max(n_iters, 1) for k, v in self._acc.items()}


# ---------------------------------------------------------------------------
# 통계: multi-seed 집계 (success rate 등 확률적 지표 전용)
# ---------------------------------------------------------------------------
def agg_seed_runs(values: list) -> dict:
    """
    여러 seed에서 얻은 값(예: seed별 success_rate_pct)의 평균/표준편차/95% CI를 계산한다.
    - n=3(권장 최소)~5(권장) 기준. t-분포 근사 대신 정규근사(1.96*sem)를 쓴다(소표본 근사이나 실무적으로 충분).
    """
    import math
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
        sem = std / math.sqrt(n)
        ci95 = 1.96 * sem
    else:
        std, ci95 = 0.0, 0.0
    return {
        "n_seeds": n,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci95": round(ci95, 3),
        "ci95_low": round(mean - ci95, 3),
        "ci95_high": round(mean + ci95, 3),
        "values": values,
    }