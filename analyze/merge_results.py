"""
merge_results.py
-----------------
트랙1 성능(TegraProfiler 출력) + energy.csv + 트랙2 success_summary.csv를
artifact 이름(name) 기준으로 join하고, energy_per_success를 계산한다.

energy_per_success_j = (episode당 평균 net_energy_J) / (success_rate/100)
  - 여기서는 트랙1 energy.csv의 energy_j_net(measure_iters 구간 총량)을
    "1 action당" 에너지로 보고, 직접 success rate로 정규화한다.
  - 정밀하게 하려면 트랙2 rollout 중 episode별 에너지를 따로 재는 것이 이상적이나
    (설계서 §5 전제), 이 스크립트는 트랙1 energy_mj_per_action을 대체 근사치로 사용한다.

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
