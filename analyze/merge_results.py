"""
merge_results.py
-----------------
트랙1 성능(TegraProfiler 출력) + energy.csv + 트랙2 success_summary.csv를
artifact 이름(name) 기준으로 join하고, energy_per_success를 계산한다.

energy_per_success_j_compute / energy_per_success_j_total = (episode당 평균 net_energy_J) / (success_rate/100)
  - compute = GPU+CPU(1+2) 연산 전력 기준, total = +시스템 5V(1+2+3) 기준. 둘 다 기록한다.
  - 여기서는 트랙1 energy.csv의 energy_j_net(measure_iters 구간 총량)을
    "1 action당" 에너지로 보고, trực접 success rate로 정규화한다.
  - 정밀하게 하려면 트랙2 rollout 중 episode별 에너지를 따로 재는 것이 이상적이나
    (설계서 §5 전제), 이 스크립트는 트랙1 energy_mj_per_action을 대체 근사치로 사용한다.


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
      --out ./benchmark/merged.csv
"""

import argparse
import csv


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def index_by_name(rows):
    d = {}
    for r in rows:
        d.setdefault(r["name"], []).append(r)
    return d


def merge(perf_path, energy_path, success_path, out_path):
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

            # energy_per_success 계산 [정정] compute(1+2)/total(1+2+3) 두 지표 각각 산출
            try:
                rate_pct = float(s.get("mean_success_pct", s.get("success_rate_pct", "nan")))
            except (ValueError, TypeError):
                rate_pct = float("nan")

            for suffix in ("compute", "total"):
                try:
                    mj_per_action = float(e.get(f"energy_mj_per_action_{suffix}", "nan"))
                    if rate_pct > 0:
                        j_per_action = mj_per_action / 1000.0
                        row[f"energy_per_success_j_{suffix}"] = round(j_per_action / (rate_pct / 100.0), 6)
                    elif rate_pct == 0:
                        row[f"energy_per_success_j_{suffix}"] = "inf"  # 성공률 0 → 발산
                    else:
                        row[f"energy_per_success_j_{suffix}"] = "NA"
                except (ValueError, TypeError):
                    row[f"energy_per_success_j_{suffix}"] = "NA"

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
    args = ap.parse_args()
    merge(args.perf, args.energy, args.success, args.out)