"""
merge_results.py
-----------------
트랙1 성능(TegraProfiler 출력) + energy.csv + 트랙2 success_summary.csv를
artifact 이름(name) 기준으로 join하고, energy_per_success를 계산한다.

energy_per_success_j = (episode당 평균 net_energy_J) / (success_rate/100)
  - 여기서는 트랙1 energy.csv의 energy_j_net(measure_iters 구간 총량)을
    "1 action당" 에너지로 보고, 직접 success rate로 정규화한다.
  - 이 값은 사후 근사치다(정적 200회 반복, 더미 입력 기준이라 episode마다 실제 action 수가
    다르다는 걸 반영하지 못한다). 그래도 삭제하지 않고 남겨둔다 — 아래 energy_per_success_j_direct
    (트랙2 실측치, run_libero_remote.py가 episode별로 직접 잰 값)와 나란히 비교하면
    근사치가 실측과 얼마나 벗어나는지 자체가 흥미로운 검증 포인트가 되기 때문이다.

latency_consistency_check
  - 트랙1(통제된 단독 측정)의 latency와 트랙2(closed-loop 중 server_infer_ms)가
    서로 일치하는지 자동 비교해, 측정 방법론 자체의 신뢰도를 검증한다.
  - 임계값(기본 15%)을 벗어나면 "diverged"로 표시하고 경고를 출력한다(열 스로틀링,
    장시간 세션 열화, 워밍업 부족 등을 의심할 신호).

실행 예)
  python -m analyze.merge_results \
      --perf ./benchmark/<tegraprofiler_output>.csv \
      --energy ./benchmark/energy.csv \
      --success ./benchmark/success_summary.csv \
      --out ./benchmark/merged.csv \
      --consistency-threshold-pct 15
"""

import argparse
import csv

# Track1(단독 측정) latency 컬럼 후보. 실제 TegraProfiler CSV의 정확한 컬럼명이 배포마다
# 다를 수 있어 하드코딩 대신 후보 목록에서 순서대로 탐색한다. 맞는 이름이 없으면
# --track1-latency-col로 직접 지정한다.
TRACK1_LATENCY_COL_CANDIDATES = [
    "lat_median_ms", "latency_ms_median", "median_latency_ms", "latency_ms_p50",
    "lat_mean_ms", "latency_ms_mean", "mean_latency_ms", "latency_ms",
]
DEFAULT_CONSISTENCY_THRESHOLD_PCT = 15.0


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def index_by_name(rows):
    d = {}
    for r in rows:
        d.setdefault(r["name"], []).append(r)
    return d


def _find_track1_latency(perf_row: dict, track1_latency_col: str = None):
    """perf_row(트랙1 CSV 한 행)에서 latency(ms) 값을 찾는다.
    --track1-latency-col로 명시했으면 그것만 쓰고, 아니면 후보 목록에서 첫 매치를 쓴다.
    """
    if track1_latency_col:
        v = perf_row.get(track1_latency_col)
        return (track1_latency_col, v) if v not in (None, "") else (track1_latency_col, None)
    for col in TRACK1_LATENCY_COL_CANDIDATES:
        v = perf_row.get(col)
        if v not in (None, ""):
            return col, v
    return None, None


def latency_consistency_check(perf_row: dict, success_row: dict,
                              threshold_pct: float = DEFAULT_CONSISTENCY_THRESHOLD_PCT,
                              track1_latency_col: str = None) -> dict:
    """
    트랙1(통제된 단독 측정)과 트랙2(closed-loop 중 실측 server_infer_ms)가 서로 일치하는지
    비교한다. 두 트랙은 서로 다른 측정 방법론(정적 반복 vs closed-loop)이므로, 방법론 자체가
    믿을 만한지 검증하는 자기 점검 지표로 쓴다.

    반환: track1_latency_ms, track2_server_infer_ms, latency_diff_ms, latency_diff_pct,
          latency_consistency("consistent"/"diverged"/"NA") 5개 키의 dict.
    """
    col, track1_v = _find_track1_latency(perf_row, track1_latency_col)
    track2_v = success_row.get("server_infer_ms_mean")

    try:
        track1_ms = float(track1_v)
        track2_ms = float(track2_v)
        diff_ms = track2_ms - track1_ms
        diff_pct = (diff_ms / track1_ms * 100.0) if track1_ms != 0 else float("nan")
        consistency = "consistent" if abs(diff_pct) <= threshold_pct else "diverged"
        return {
            "track1_latency_ms": round(track1_ms, 4),
            "track2_server_infer_ms": round(track2_ms, 4),
            "latency_diff_ms": round(diff_ms, 4),
            "latency_diff_pct": round(diff_pct, 2),
            "latency_consistency": consistency,
        }
    except (TypeError, ValueError):
        return {
            "track1_latency_ms": "NA",
            "track2_server_infer_ms": "NA",
            "latency_diff_ms": "NA",
            "latency_diff_pct": "NA",
            "latency_consistency": "NA",
        }


def merge(perf_path, energy_path, success_path, out_path,
         consistency_threshold_pct=DEFAULT_CONSISTENCY_THRESHOLD_PCT,
         track1_latency_col=None):
    perf = index_by_name(read_csv(perf_path)) if perf_path else {}
    energy = index_by_name(read_csv(energy_path)) if energy_path else {}
    success = index_by_name(read_csv(success_path)) if success_path else {}

    names = set(perf) | set(energy) | set(success)
    out_rows = []

    for name in sorted(names):
        p = perf.get(name, [{}])[-1]
        e = energy.get(name, [{}])[-1]
        # success는 suite별로 여러 행일 수 있음 → 각각 별도 output row
        s_rows = success.get(name, [{}])

        for s in s_rows:
            row = {"name": name}
            row.update({f"perf_{k}": v for k, v in p.items() if k != "name"})
            row.update({f"energy_{k}": v for k, v in e.items() if k != "name"})
            row.update({f"success_{k}": v for k, v in s.items() if k != "name"})

            # energy_per_success 계산
            try:
                mj_per_action = float(e.get("energy_mj_per_action", "nan"))
                rate_pct = float(s.get("mean_success_pct", s.get("success_rate_pct", "nan")))
                if rate_pct > 0:
                    j_per_action = mj_per_action / 1000.0
                    row["energy_per_success_j"] = round(j_per_action / (rate_pct / 100.0), 6)
                else:
                    row["energy_per_success_j"] = "inf"  # 성공률 0 → 발산(설계서 §4 주의사항)
            except (ValueError, TypeError):
                row["energy_per_success_j"] = "NA"

            # 트랙2 직접실측 energy_per_success(J) — run_libero_remote.py가 episode별로 잰 값.
            # 근사치(energy_per_success_j, 트랙1 기반)와 나란히 두어 괴리를 눈으로 검증한다.
            row["energy_per_success_j_direct"] = s.get("energy_per_success_j_direct", "NA")

            # 트랙1 예측 vs 트랙2 실측 latency 일치도 검증
            row.update(latency_consistency_check(
                p, s, threshold_pct=consistency_threshold_pct,
                track1_latency_col=track1_latency_col,
            ))
            if row["latency_consistency"] == "diverged":
                print(f"[merge][경고] {name}/{s.get('suite', '?')}: 트랙1-트랙2 latency 불일치 "
                      f"({row['track1_latency_ms']}ms vs {row['track2_server_infer_ms']}ms, "
                      f"{row['latency_diff_pct']}% > 임계값 {consistency_threshold_pct}%). "
                      f"열 스로틀링, 장시간 세션 열화, 워밍업 부족 등을 확인해보세요.")

            out_rows.append(row)

    if not out_rows:
        print("[merge] 병합할 데이터가 없습니다.")
        return

    fieldnames = sorted({k for r in out_rows for k in r})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[merge] {len(out_rows)}행 → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", default=None, help="TegraProfiler 성능 CSV 경로")
    ap.add_argument("--energy", default="./benchmark/energy.csv")
    ap.add_argument("--success", default="./benchmark/success_summary.csv")
    ap.add_argument("--out", default="./benchmark/merged.csv")
    ap.add_argument("--consistency-threshold-pct", type=float,
                    default=DEFAULT_CONSISTENCY_THRESHOLD_PCT,
                    help="트랙1-트랙2 latency 일치 판정 임계값(%%, 기본 15)")
    ap.add_argument("--track1-latency-col", default=None,
                    help="트랙1 CSV의 latency(ms) 컬럼명을 직접 지정(기본: 후보 목록에서 자동탐색)")
    args = ap.parse_args()
    merge(args.perf, args.energy, args.success, args.out,
         consistency_threshold_pct=args.consistency_threshold_pct,
         track1_latency_col=args.track1_latency_col)
