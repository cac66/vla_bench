# VLA 경량화 벤치마크 코드 정리

Jetson(모델) + host(시뮬) 구성에서 OpenVLA·SmolVLA artifact를 측정하는 코드 모음이다.

## 개정 사항 (measurement_extension_spec.md 반영)

- **Energy**: `energy.py` — sysfs(`/sys/class/hwmon/hwmon1`, VDD_IN 레일) 직접 폴링. power 노드가 없어 전압×전류로 계산. idle-baseline(30s) 차감한 순수 에너지 + 총 에너지 둘 다 산출, `mJ/action`으로 정규화. `bench_common.run_artifact`가 측정 구간에 자동 연동.
- **Suite별 성공률 분해**: `run_libero_remote.run_all_suites` — 여러 suite를 순회해 결과를 나란히 기록.
- **Action smoothness**: `smoothness.py` — LDLJ + JerkRMS 2종(확정), 성공 episode만 계산.
- **제어주파수-성공률 곡선**: `run_libero_remote.py --target-hz-sweep` — 30/15/10/6/3 Hz로 인위적 지연 주입.
- **Success rate 통계**: `agg_seed_runs`(bench_common.py) — 기본 3 seed 반복, 평균/표준편차/95% CI 보고.
- **Energy per success**: `analyze/merge_results.py` — 트랙1 energy.csv + 트랙2 success_summary.csv를 artifact 이름으로 join해 계산.

## 측정 철학 (코드가 강제하는 원칙)

- **모델 추론은 항상 Jetson.** 성능은 Jetson 단독, 정확도는 서버로 Jetson에 위임.
- **시뮬은 host.** LIBERO는 host에서 돌고, 모델은 네트워크 너머 Jetson에서 돈다.
- **측정을 3층으로 분리한다.**

| 층 | 무엇 | 시뮬 | 실행 위치 | 코드 |
|----|------|------|-----------|------|
| 트랙 1 | latency·power·memory | 불필요 | Jetson 단독 | `bench_*.py --mode latency` |
| 스크리닝 | open-loop action MSE | 불필요 | Jetson 단독 | `bench_*.py --mode mse` |
| 트랙 2 | task success rate | 필요 | host 시뮬 + Jetson 모델 | `serving/` |

## 파일 구조

```
vla_bench/
├── bench_common.py          # 하네스: ArtifactSpec, run_artifact(트랙1), screen_mse(스크리닝), CSV
├── bench_openvla.py         # OpenVLA artifact(A0~A5) + 로더/전처리/추론 공유 + CLI
├── bench_smolvla.py         # SmolVLA artifact(B0~B3) + 로더/전처리/추론 공유 + CLI
├── serving/
│   ├── protocol.py          # ZMQ+pickle 관측/행동 직렬화
│   ├── policy_server.py     # [Jetson] artifact 로드 → action 서빙
│   ├── policy_client.py     # [host] RemotePolicy (drop-in)
│   └── run_libero_remote.py # [host] LIBERO closed-loop rollout(원격 정책)
└── benchmark/               # 결과 CSV 출력 루트
```

## 전체 실행 흐름 (5단계)

앞서 정한 "성능 먼저 → MSE로 거르고 → 살아남은 것만 시뮬" 순서를 그대로 코드화했다.

1. **트랙 1 (전체 artifact 성능)** — Jetson에서 단독 실행.
   ```bash
   # AGX Orin
   python bench_openvla.py --mode latency
   # Orin Nano
   python bench_smolvla.py --mode latency
   ```
2. **스크리닝 (MSE)** — 시뮬 없이 관측 집합으로 quantization 편차 확인.
   ```bash
   python bench_openvla.py --mode mse --eval-npz ./data/openvla_eval.npz
   python bench_smolvla.py --mode mse --repo HuggingFaceVLA/libero
   ```
   → `mse_vs_fp16`이 큰 artifact를 트랙 2 후보에서 제외한다.
3. **후보 선별** — 성능 CSV + MSE CSV로 Pareto 후보만 남긴다(수작업/스크립트).
4. **트랙 2 (선별된 소수 정확도)** — Jetson에 서버, host에 시뮬.
   ```bash
   # (Jetson) 측정할 artifact를 서빙
   python -m serving.policy_server --model openvla --artifact A2_int4 --port 5555
   # (host) 시뮬 rollout — 모델은 Jetson에서 돈다
   python -m serving.run_libero_remote --server <JETSON_IP>:5555 \
          --suite libero_spatial --episodes 50 --artifact A2_int4
   ```
5. **병합·분석** — artifact 이름을 key로 성능/MSE/성공률 CSV를 합쳐 trade-off·Pareto 작성.

## 각 파일 설명

- **bench_common.py** — 측정 loop를 한 곳에 고정한다. `run_artifact`는 주어진 TegraProfiler API를 그대로 호출(warmup→measure→`record_latency`→선택적 breakdown/accuracy)하고, artifact마다 `free_model`로 GPU를 비워 오염을 막는다. `screen_mse`는 기준(FP16)을 먼저 돌려 그 예측을 기준으로 다른 artifact의 `mse_vs_fp16`(및 선택적 `mse_vs_gt`)을 계산해 CSV로 남긴다.
- **bench_openvla.py / bench_smolvla.py** — 한 모델의 로딩·전처리·추론 지식을 **세 용도(latency 더미 입력 / MSE 실제 관측 / serving 네트워크 관측)가 공유**하도록 `preprocess_*`·`predict_*`로 통일했다. `LOADERS` dict가 artifact 이름→로더를 제공해 서버가 그대로 재사용한다.
- **serving/protocol.py** — 관측/행동을 ZMQ REQ/REP로 주고받는 얇은 직렬화(사설 LAN 전제 pickle).
- **serving/policy_server.py** — Jetson에서 `LOADERS[artifact]()`로 모델을 올리고, 관측을 받아 `predict_*`로 추론해 action을 반환한다. **여기가 "모델은 Jetson" 원칙의 실체**다.
- **serving/policy_client.py** — host용 `RemotePolicy.predict(obs)`. 왕복 시간(rtt)과 서버 보고 추론시간(server_infer_ms)을 함께 로깅해 네트워크와 순수 추론을 분리한다.
- **serving/run_libero_remote.py** — LIBERO를 host에서 돌리며 매 step 관측을 원격 정책에 넘겨 action을 받는다. 성공률을 CSV로 남긴다.

## 결과 CSV 병합

세 산출물은 artifact 이름(및 device/precision/technique)을 공통 key로 갖는다.
- 트랙 1: TegraProfiler가 쓰는 성능 CSV (lat_*, power, memory 등)
- 스크리닝: `mse_screen_*.csv` (mse_vs_fp16, mse_vs_gt)
- 트랙 2: `success_remote.csv` (success_rate_pct, rtt_ms_mean, server_infer_ms_mean)

이름 key로 join하면 artifact별 (속도 / 정확도 / 메모리 / 전력) 한 행이 완성되고, 이것으로 Pareto frontier를 그린다.

## 실행 전 확인·교체 지점 (TODO)

1. **SmolVLA flow step attribute** (`policy.config.num_steps`) — 설치 LeRobot 버전에서 실제 이름 확인. B1의 핵심.
2. **SmolVLA chunk 재생성** — `policy.reset()` 메서드명/동작 확인(캐시된 step은 ~0ms라 측정 왜곡).
3. **OpenVLA submodule 이름** — breakdown용 `vision_backbone`/`language_model`을 `print(model)`로 조정.
4. **A3 AWQ / A5 token_prune** — AWQ는 사전 빌드 checkpoint 경로, token_prune은 실제 감축 모듈 주입 필요.
5. **MSE 관측 소스** — OpenVLA는 `openvla_eval.npz`(images/instructions/actions)를 미리 생성, SmolVLA는 `HuggingFaceVLA/libero`에서 로드.
6. **LIBERO 성공 판정** — 엄밀한 재현은 공식 `run_libero_eval.py`의 정책 호출부를 `RemotePolicy`로 교체하는 방식 권장.
7. **의존성** — 서버/클라이언트는 `pyzmq` 필요(`pip install pyzmq`).
```
