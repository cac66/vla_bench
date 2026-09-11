"""
tegra_profiler.py
VLA inference benchmark profiler for NVIDIA Jetson (tegrastats-based).

Outputs per run:
  ./results_summary.csv            — append-mode aggregate (one row per run)
  ./runs/<run_id>_timeseries.csv   — raw per-sample time series
  ./runs/<run_id>.md               — human-readable benchmark report

Usage — context manager:
    from tegra_profiler import TegraProfiler
    import torch, time

    prof = TegraProfiler(
        device="orin_nano_8gb", model="smolvla",
        precision="fp16", runtime="pytorch", technique="none",
        warmup_iters=30, measure_iters=200,
    )
    with prof:
        for _ in range(prof.warmup_iters):
            _ = model(inputs)
        torch.cuda.synchronize()
        lat_ms = []
        for _ in range(prof.measure_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(inputs)
            torch.cuda.synchronize()
            lat_ms.append((time.perf_counter() - t0) * 1000)
        prof.record_latency(lat_ms)
        # optional:
        # prof.record_latency_breakdown(vision_ms=12.1, backbone_ms=8.4, action_ms=3.0)
        # prof.record_accuracy(success_rate_pct=87.5, action_mse=0.042, eval_env="Jetson")

Usage — parse existing log:
    python tegra_profiler.py --parse raw.log --run-id my_run --device orin_nano_8gb
"""

#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import subprocess
import threading
import time
from datetime import datetime

# --------------------------------------------------------------------------- #
# Warning thresholds — adjust here without touching logic                      #
# --------------------------------------------------------------------------- #
WARN_GPU_LOW_PCT    = 80    # below this → "GPU underutilized"
WARN_SWAP_THRESH_MB = 0     # above this → "SWAP 발생"
WARN_TJ_THRESH_C    = 95   # at or above → "thermal throttling 의심"

# --------------------------------------------------------------------------- #
# Fixed column schema                                                          #
# --------------------------------------------------------------------------- #
SUMMARY_COLS = [
    "run_id", "timestamp", "device", "model",
    "precision", "runtime", "technique",
    "replan_interval", "action_chunk_size", "seq_len",
    "nvpmodel_mode", "jetson_clocks",
    "warmup_iters", "measure_iters",
    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
    "lat_std_ms", "throughput_hz",
    "lat_vision_ms", "lat_backbone_ms", "lat_action_ms",
    "ram_used_peak_mb", "ram_used_delta_mb", "ram_total_mb",
    "swap_used_peak_mb", "lfb_mb",
    "gpu_load_mean_pct", "gpu_load_min_pct", "emc_util_mean_pct",
    "cpu_util_mean_pct", "cpu_util_max_pct",
    "temp_tj_max_c", "temp_soc_max_c",
    "success_rate_pct", "action_mse",
    "notes",
]

TIMESERIES_COLS = [
    "t_ms",
    "ram_used_mb", "swap_used_mb", "lfb_mb",
    "cpu_mean_pct", "cpu_max_pct",
    "gpu_load_pct", "emc_util_pct",
    "temp_tj_c", "temp_soc0_c", "temp_soc1_c", "temp_soc2_c",
]

# --------------------------------------------------------------------------- #
# 1. PARSING                                                                   #
# --------------------------------------------------------------------------- #

_RE = {
    "ram":    re.compile(r"RAM (\d+)/(\d+)MB"),
    "lfb":    re.compile(r"\(lfb (\d+)x(\d+)MB\)"),
    "swap":   re.compile(r"SWAP (\d+)/(\d+)MB"),
    "cpu":    re.compile(r"CPU \[([^\]]+)\]"),
    "emc":    re.compile(r"EMC_FREQ (\d+)%"),
    "gr3d":   re.compile(r"GR3D_FREQ (\d+)%"),
    "temp":   re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C"),
    "core":   re.compile(r"(\d+)%@(\d+)"),
    "epoch":  re.compile(r"^(\d+\.\d+)\s"),
    "ts":     re.compile(r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})"),
}

TEMP_INVALID = -256.0   # disabled sensor marker


def _extract_time(line):
    m = _RE["epoch"].match(line)
    if m:
        ep = float(m.group(1))
        return ep, datetime.fromtimestamp(ep).isoformat(timespec="seconds")
    m = _RE["ts"].search(line)
    if m:
        dt = datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")
        return dt.timestamp(), dt.isoformat(timespec="seconds")
    return None, None


def parse_line(line):
    """Parse one tegrastats line → flat dict, or None if no RAM field present."""
    line = line.strip()
    if "RAM " not in line:
        return None
    row = {}

    epoch, iso = _extract_time(line)
    row["epoch"] = epoch
    row["timestamp"] = iso

    m = _RE["ram"].search(line)
    if m:
        used, total = int(m.group(1)), int(m.group(2))
        row["ram_used_mb"] = used
        row["ram_total_mb"] = total

    m = _RE["lfb"].search(line)
    if m:
        # lfb NxMMB → largest free block = M MB
        row["lfb_block_mb"] = int(m.group(2))

    m = _RE["swap"].search(line)
    if m:
        row["swap_used_mb"] = int(m.group(1))

    m = _RE["cpu"].search(line)
    if m:
        cores = _RE["core"].findall(m.group(1))
        utils = [int(u) for u, _ in cores]
        if utils:
            row["cpu_avg_pct"] = round(sum(utils) / len(utils), 2)
            row["cpu_max_pct"] = max(utils)

    m = _RE["emc"].search(line)
    if m:
        row["emc_pct"] = int(m.group(1))

    m = _RE["gr3d"].search(line)
    if m:
        row["gpu_pct"] = int(m.group(1))

    for name, val in _RE["temp"].findall(line):
        v = float(val)
        if v <= TEMP_INVALID:
            continue
        row[f"temp_{name.lower()}_c"] = v

    return row


def parse_log(path, t_start=None, t_end=None):
    """Read a tegrastats log → list of row dicts, optionally time-windowed."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            row = parse_line(line)
            if row is None:
                continue
            if t_start is not None:
                ep = row.get("epoch")
                if ep is None or ep < t_start or ep > t_end:
                    continue
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 2. STATISTICS                                                                #
# --------------------------------------------------------------------------- #

def _col(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def _percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def latency_stats(lat_ms):
    """Compute distribution metrics from a list of per-inference latencies (ms)."""
    if not lat_ms:
        return {}
    n = len(lat_ms)
    mean = sum(lat_ms) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in lat_ms) / n)
    return {
        "lat_mean_ms":   round(mean, 3),
        "lat_p50_ms":    round(_percentile(lat_ms, 50), 3),
        "lat_p95_ms":    round(_percentile(lat_ms, 95), 3),
        "lat_p99_ms":    round(_percentile(lat_ms, 99), 3),
        "lat_std_ms":    round(std, 3),
        "throughput_hz": round(1000.0 / mean, 3) if mean > 0 else "NA",
    }


def summarize(rows):
    """Aggregate resource statistics from parsed tegrastats rows."""
    if not rows:
        return {}
    s = {}

    ram = _col(rows, "ram_used_mb")
    if ram:
        s["ram_used_peak_mb"]  = max(ram)
        s["ram_used_delta_mb"] = max(ram) - ram[0]
    totals = _col(rows, "ram_total_mb")
    if totals:
        s["ram_total_mb"] = totals[0]

    swap = _col(rows, "swap_used_mb")
    s["swap_used_peak_mb"] = max(swap) if swap else 0

    lfb = _col(rows, "lfb_block_mb")
    if lfb:
        s["lfb_mb"] = min(lfb)  # worst-case largest-free-block across run

    gpu = _col(rows, "gpu_pct")
    if gpu:
        s["gpu_load_mean_pct"] = round(sum(gpu) / len(gpu), 2)
        s["gpu_load_min_pct"]  = min(gpu)

    emc = _col(rows, "emc_pct")
    if emc:
        s["emc_util_mean_pct"] = round(sum(emc) / len(emc), 2)

    cpu_avg = _col(rows, "cpu_avg_pct")
    cpu_max = _col(rows, "cpu_max_pct")
    if cpu_avg:
        s["cpu_util_mean_pct"] = round(sum(cpu_avg) / len(cpu_avg), 2)
    if cpu_max:
        s["cpu_util_max_pct"] = max(cpu_max)

    tj = _col(rows, "temp_tj_c")
    if tj:
        s["temp_tj_max_c"] = max(tj)

    soc_peak = []
    for k in ("temp_soc0_c", "temp_soc1_c", "temp_soc2_c"):
        v = _col(rows, k)
        if v:
            soc_peak.append(max(v))
    if soc_peak:
        s["temp_soc_max_c"] = max(soc_peak)

    return s


# --------------------------------------------------------------------------- #
# 3. OUTPUT WRITERS                                                            #
# --------------------------------------------------------------------------- #

def _na(v):
    """Return v if meaningful, otherwise the string 'NA'."""
    return v if (v is not None and v != "") else "NA"


def _build_summary_row(run_config, lat_stats_dict, tegra_summary,
                       breakdown=None, accuracy=None):
    """Merge all inputs into a dict aligned to SUMMARY_COLS; gaps → 'NA'."""
    row = {col: "NA" for col in SUMMARY_COLS}
    for src in (run_config, lat_stats_dict, tegra_summary):
        for k, v in src.items():
            if k in row:
                row[k] = _na(v)
    if breakdown:
        row["lat_vision_ms"]   = _na(breakdown.get("vision_ms"))
        row["lat_backbone_ms"] = _na(breakdown.get("backbone_ms"))
        row["lat_action_ms"]   = _na(breakdown.get("action_ms"))
    if accuracy:
        row["success_rate_pct"] = _na(accuracy.get("success_rate_pct"))
        row["action_mse"]       = _na(accuracy.get("action_mse"))
    return row


def write_summary_csv(row, path):
    """Append one summary row; writes header if file is new or empty."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_timeseries_csv(rows, run_id, runs_dir):
    """Write per-sample timeseries to runs/<run_id>_timeseries.csv."""
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{run_id}_timeseries.csv")

    epochs = _col(rows, "epoch")
    t0 = epochs[0] if epochs else None

    out_rows = []
    for r in rows:
        ep = r.get("epoch")
        t_ms = round((ep - t0) * 1000, 1) if (ep is not None and t0 is not None) else "NA"
        out_rows.append({
            "t_ms":        t_ms,
            "ram_used_mb": _na(r.get("ram_used_mb")),
            "swap_used_mb": _na(r.get("swap_used_mb")),
            "lfb_mb":      _na(r.get("lfb_block_mb")),
            "cpu_mean_pct": _na(r.get("cpu_avg_pct")),
            "cpu_max_pct": _na(r.get("cpu_max_pct")),
            "gpu_load_pct": _na(r.get("gpu_pct")),
            "emc_util_pct": _na(r.get("emc_pct")),
            "temp_tj_c":   _na(r.get("temp_tj_c")),
            "temp_soc0_c": _na(r.get("temp_soc0_c")),
            "temp_soc1_c": _na(r.get("temp_soc1_c")),
            "temp_soc2_c": _na(r.get("temp_soc2_c")),
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TIMESERIES_COLS)
        w.writeheader()
        w.writerows(out_rows)
    return path


def _build_warnings(tegra_summary):
    warns = []
    gpu_mean = tegra_summary.get("gpu_load_mean_pct")
    if gpu_mean is not None and gpu_mean < WARN_GPU_LOW_PCT:
        warns.append(
            f"GPU underutilized (mean {gpu_mean}% < {WARN_GPU_LOW_PCT}%) — 병목 의심")
    swap_peak = tegra_summary.get("swap_used_peak_mb", 0)
    if swap_peak > WARN_SWAP_THRESH_MB:
        warns.append(f"SWAP 발생 (peak {swap_peak} MB) — 메모리 압박")
    tj_max = tegra_summary.get("temp_tj_max_c")
    if tj_max is not None and tj_max >= WARN_TJ_THRESH_C:
        warns.append(
            f"Thermal throttling 의심 (tj max {tj_max}°C ≥ {WARN_TJ_THRESH_C}°C)")
    return warns


def write_benchmark_report(run_id, run_config, lat_stats_dict, tegra_summary,
                           breakdown=None, accuracy=None, runs_dir="runs"):
    """Write a human-readable Markdown report to runs/<run_id>.md."""
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{run_id}.md")
    warns = _build_warnings(tegra_summary)

    lines = []
    a = lines.append

    a(f"## Run {run_id}\n")

    # --- Metadata ---
    a("### Run Metadata\n")
    a("| field | value |")
    a("|---|---|")
    for field in ("device", "model", "precision", "runtime", "technique",
                  "replan_interval", "action_chunk_size", "seq_len",
                  "nvpmodel_mode", "jetson_clocks", "warmup_iters", "measure_iters"):
        a(f"| {field} | {_na(run_config.get(field))} |")
    a("")

    # --- Latency ---
    a("### Latency\n")
    a("| metric | value |")
    a("|---|---|")
    for k in ("lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
              "lat_std_ms", "throughput_hz"):
        a(f"| {k} | {_na(lat_stats_dict.get(k))} |")
    a("")

    # --- Component breakdown (optional) ---
    if breakdown:
        a("### Component Breakdown\n")
        a("| component | latency (ms) |")
        a("|---|---|")
        a(f"| vision   | {_na(breakdown.get('vision_ms'))} |")
        a(f"| backbone | {_na(breakdown.get('backbone_ms'))} |")
        a(f"| action   | {_na(breakdown.get('action_ms'))} |")
        a("")

    # --- Resource ---
    a("### Resource Usage\n")
    a("| metric | value |")
    a("|---|---|")
    a(f"| RAM peak (MB)      | {_na(tegra_summary.get('ram_used_peak_mb'))} |")
    a(f"| RAM delta (MB)     | {_na(tegra_summary.get('ram_used_delta_mb'))} |")
    a(f"| SWAP peak (MB)     | {_na(tegra_summary.get('swap_used_peak_mb'))} |")
    a(f"| LFB min (MB)       | {_na(tegra_summary.get('lfb_mb'))} |")
    a(f"| GPU load mean (%)  | {_na(tegra_summary.get('gpu_load_mean_pct'))} |")
    a(f"| GPU load min (%)   | {_na(tegra_summary.get('gpu_load_min_pct'))} |")
    a(f"| EMC util mean (%)  | {_na(tegra_summary.get('emc_util_mean_pct'))} |")
    a(f"| CPU util mean (%)  | {_na(tegra_summary.get('cpu_util_mean_pct'))} |")
    a(f"| CPU util max (%)   | {_na(tegra_summary.get('cpu_util_max_pct'))} |")
    a("")

    # --- Thermal ---
    a("### Thermal\n")
    a("| sensor | max (°C) |")
    a("|---|---|")
    a(f"| tj      | {_na(tegra_summary.get('temp_tj_max_c'))} |")
    a(f"| SOC max | {_na(tegra_summary.get('temp_soc_max_c'))} |")
    tj_max = tegra_summary.get("temp_tj_max_c")
    if tj_max is not None and tj_max >= WARN_TJ_THRESH_C:
        a(f"\n> Throttling 의심: tj max {tj_max}°C — timeseries에서 해당 구간 확인 권장")
    else:
        a("\n> Thermal throttling 의심 구간 없음")
    a("")

    # --- Accuracy (optional) ---
    a("### Accuracy\n")
    if accuracy:
        env = accuracy.get("eval_env", "NA")
        a("| metric | value | eval env |")
        a("|---|---|---|")
        a(f"| success_rate_pct | {_na(accuracy.get('success_rate_pct'))} | {env} |")
        a(f"| action_mse       | {_na(accuracy.get('action_mse'))} | {env} |")
    else:
        a("_별도 평가 결과 없음_")
    a("")

    # --- Warnings ---
    a("### Warnings\n")
    if warns:
        for w in warns:
            a(f"- {w}")
    else:
        a("_경고 없음_")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 4. DEVICE METADATA (best-effort, never crashes)                             #
# --------------------------------------------------------------------------- #

def read_device_meta():
    """Query nvpmodel and jetson_clocks from the device; returns partial dict."""
    meta = {}
    try:
        out = subprocess.check_output(
            ["nvpmodel", "-q"], text=True,
            stderr=subprocess.DEVNULL, timeout=5)
        meta["nvpmodel_mode"] = " ".join(out.split())
    except Exception:
        pass
    try:
        subprocess.check_output(
            ["jetson_clocks", "--show"], text=True,
            stderr=subprocess.DEVNULL, timeout=5)
        meta["jetson_clocks"] = "on"
    except subprocess.CalledProcessError:
        meta["jetson_clocks"] = "off"
    except Exception:
        pass
    return meta


# --------------------------------------------------------------------------- #
# 5. PROFILER (context manager)                                                #
# --------------------------------------------------------------------------- #

class TegraProfiler:
    """
    Captures tegrastats during the inference window and writes three outputs:
      results_summary.csv, runs/<run_id>_timeseries.csv, runs/<run_id>.md

    record_latency(lat_ms)             — list of per-inference latencies in ms
    record_latency_breakdown(...)      — optional vision/backbone/action breakdown
    record_accuracy(...)               — optional success_rate_pct / action_mse
    """

    def __init__(self, device="NA", model="NA", precision="NA",
                 runtime="NA", technique="none",
                 replan_interval="NA", action_chunk_size="NA", seq_len="NA",
                 nvpmodel_mode="NA", jetson_clocks="NA",
                 warmup_iters=30, measure_iters=200,
                 notes="",
                 interval_ms=100, out_root="."):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interval_ms = interval_ms
        self.runs_dir = os.path.join(out_root, "runs")
        self.summary_path = os.path.join(out_root, "results_summary.csv")

        self.run_config = {
            "run_id":           self.run_id,
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
            "device":           device,
            "model":            model,
            "precision":        precision,
            "runtime":          runtime,
            "technique":        technique,
            "replan_interval":  replan_interval,
            "action_chunk_size": action_chunk_size,
            "seq_len":          seq_len,
            "nvpmodel_mode":    nvpmodel_mode,
            "jetson_clocks":    jetson_clocks,
            "warmup_iters":     warmup_iters,
            "measure_iters":    measure_iters,
            "notes":            notes,
        }
        self.warmup_iters  = warmup_iters
        self.measure_iters = measure_iters

        self._lat_ms    = []
        self._breakdown = None
        self._accuracy  = None
        self._proc      = None
        self._reader    = None
        self._stop      = threading.Event()
        self.t_start    = None
        self.t_end      = None
        self._raw_path  = os.path.join(self.runs_dir, f"{self.run_id}_raw.log")

    def record_latency(self, lat_ms):
        """Pass measured latencies (ms) from the post-warmup inference loop."""
        self._lat_ms = list(lat_ms)

    def record_latency_breakdown(self, vision_ms=None, backbone_ms=None,
                                  action_ms=None):
        self._breakdown = {
            "vision_ms":   vision_ms,
            "backbone_ms": backbone_ms,
            "action_ms":   action_ms,
        }

    def record_accuracy(self, success_rate_pct=None, action_mse=None,
                         eval_env="NA"):
        self._accuracy = {
            "success_rate_pct": success_rate_pct,
            "action_mse":       action_mse,
            "eval_env":         eval_env,
        }

    def _pump(self, raw_file):
        for line in self._proc.stdout:
            raw_file.write(f"{time.time():.3f} {line}")
            raw_file.flush()
            if self._stop.is_set():
                break

    def __enter__(self):
        os.makedirs(self.runs_dir, exist_ok=True)
        dev_meta = read_device_meta()
        for k, v in dev_meta.items():
            if self.run_config.get(k) in ("NA", None, ""):
                self.run_config[k] = v
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            raise RuntimeError("tegrastats not found — run on a Jetson device.")
        self._raw_file = open(self._raw_path, "w")
        self._reader = threading.Thread(
            target=self._pump, args=(self._raw_file,), daemon=True)
        self._reader.start()
        self.t_start = time.time()
        return self

    def __exit__(self, *exc):
        self.t_end = time.time()
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        if self._reader:
            self._reader.join(timeout=5)
        self._raw_file.close()

        rows = parse_log(self._raw_path, self.t_start, self.t_end)

        try:
            os.remove(self._raw_path)
        except Exception:
            pass

        ts_summary  = summarize(rows)
        lat_stats_d = latency_stats(self._lat_ms)

        ts_path = write_timeseries_csv(rows, self.run_id, self.runs_dir)
        row = _build_summary_row(
            self.run_config, lat_stats_d, ts_summary,
            self._breakdown, self._accuracy)
        write_summary_csv(row, self.summary_path)
        md_path = write_benchmark_report(
            self.run_id, self.run_config, lat_stats_d, ts_summary,
            self._breakdown, self._accuracy, self.runs_dir)

        print(f"[tegra_profiler] run_id={self.run_id}")
        print(f"  summary    → {self.summary_path}")
        print(f"  timeseries → {ts_path}")
        print(f"  report     → {md_path}")
        return False  # do not suppress inference exceptions


# --------------------------------------------------------------------------- #
# 6. CLI (--parse mode for existing logs)                                      #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Parse a raw tegrastats log into benchmark outputs.")
    ap.add_argument("--parse", metavar="LOG", required=True,
                    help="path to raw tegrastats log")
    ap.add_argument("--run-id", default=None,
                    help="run identifier (default: current timestamp)")
    ap.add_argument("--device",    default="NA")
    ap.add_argument("--model",     default="NA")
    ap.add_argument("--precision", default="NA")
    ap.add_argument("--runtime",   default="NA")
    ap.add_argument("--technique", default="none")
    ap.add_argument("--notes",     default="")
    ap.add_argument("--out",       default=".",
                    help="output root directory (default: current directory)")
    args = ap.parse_args()

    run_id    = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir  = os.path.join(args.out, "runs")
    sum_path  = os.path.join(args.out, "results_summary.csv")

    rows = parse_log(args.parse)
    ts_summary  = summarize(rows)
    lat_stats_d = {}

    run_config = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device":    args.device,
        "model":     args.model,
        "precision": args.precision,
        "runtime":   args.runtime,
        "technique": args.technique,
        "notes":     args.notes,
    }
    run_config.update(read_device_meta())

    ts_path = write_timeseries_csv(rows, run_id, runs_dir)
    row = _build_summary_row(run_config, lat_stats_d, ts_summary)
    write_summary_csv(row, sum_path)
    md_path = write_benchmark_report(
        run_id, run_config, lat_stats_d, ts_summary, runs_dir=runs_dir)

    print(f"[tegra_profiler] parsed {len(rows)} rows | run_id={run_id}")
    print(f"  summary    → {sum_path}")
    print(f"  timeseries → {ts_path}")
    print(f"  report     → {md_path}")


if __name__ == "__main__":
    main()
