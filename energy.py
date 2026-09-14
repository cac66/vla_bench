"""
energy.py
---------
sysfs(INA3221 hwmon) 직접 폴링 기반 에너지 측정.

[정정, 중요] AGX Orin의 실제 레일 구성 (NVIDIA 공식 문서 기준)
  hwmon 경로 : /sys/class/hwmon/hwmon1  (I2C 0x40 칩)
  in1 = VDD_GPU_SOC   (GPU+SOC 전력)
  in2 = VDD_CPU_CV    (CPU+CV 전력)
  in3 = VIN_SYS_5V0   (시스템 5V I/O 레일 — HDMI/USB/eMMC 등, VDDQ 포함)
  → AGX Orin에는 "VDD_IN"이라는 단일 채널이 없다(그건 Orin NX/Nano 쪽 이름이다).
    "모듈 전체 소비전력"은 여러 채널을 합산해서 만든다.

  전압대가 다르기 때문에(NVIDIA 문서 명시) in3(VIN_SYS_5V0)는 in1/in2와 별도로 다룬다.
  이에 따라 두 가지 지표를 모두 기록한다.
    - COMPUTE power = in1 + in2   (GPU+CPU 연산 전력만, SYS_VIN_HV 성격)
    - TOTAL   power = in1 + in2 + in3  (연산 + 시스템 I/O까지, 모듈 실사용 전력에 더 가까움)

  power 노드가 없으므로 in*_input(mV) * curr*_input(mA) / 1e6 = W 로 직접 계산한다.
  hwmon 인덱스·채널 배정은 기기/부팅마다 다를 수 있으니, 실기기에서 반드시
  아래로 재확인 후 HWMON_PATH를 맞춘다:
    for d in /sys/class/hwmon/hwmon*; do echo "$d -> $(cat $d/name)"; done
    for i in 1 2 3; do echo "in$i: $(cat $d/in${i}_label)"; done

측정 방법론
  - idle-baseline(BE) 측정 → active(TE) 측정 → net = TE - BE  (compute/total 각각)
  - 총 에너지(BE 미차감)와 순수 에너지(BE 차감) 둘 다 산출
  - 정규화: energy_mj_per_action (compute/total 각각)
"""

import time
import threading

HWMON_PATH = "/sys/class/hwmon/hwmon1"
RAIL_LABELS = {1: "VDD_GPU_SOC", 2: "VDD_CPU_CV", 3: "VIN_SYS_5V0"}

COMPUTE_RAILS = (1, 2)       # GPU+CPU 연산 전력
TOTAL_RAILS = (1, 2, 3)      # 연산 + 시스템 5V(I/O) 전력, 모듈 실사용에 가까움
ALL_RAILS = (1, 2, 3)        # 샘플링은 항상 전체 채널을 한 번에 뜬다


def _read_int(path: str) -> int:
    with open(path, "r") as f:
        return int(f.read().strip())


def read_all_rails_w(rails=ALL_RAILS, hwmon_path: str = HWMON_PATH) -> dict:
    """
    지정 채널들의 순간 전력(W)을 한 번에 읽어 {rail: power_w} dict로 반환한다.
    power 노드가 없으므로 전압*전류로 계산한다.
    """
    out = {}
    for r in rails:
        v_mv = _read_int(f"{hwmon_path}/in{r}_input")
        i_ma = _read_int(f"{hwmon_path}/curr{r}_input")
        out[r] = (v_mv * i_ma) / 1e6
    return out


class SysfsPowerSampler:
    """
    백그라운드 스레드로 지정 간격마다 "전체 채널(1,2,3)"을 한 번에 폴링해 적재한다.
    한 번의 샘플링으로 compute(1+2)·total(1+2+3) 두 지표를 모두 사후 계산할 수 있다.
    with 블록으로 감싸 사용: 진입 시 폴링 시작, 종료 시 폴링 중지.
    """
    def __init__(self, interval_s: float = 0.01, rails=ALL_RAILS, hwmon_path: str = HWMON_PATH):
        self.interval_s = interval_s
        self.rails = rails
        self.hwmon_path = hwmon_path
        self._samples = []          # [(t_rel_s, {rail: power_w}), ...]
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                powers = read_all_rails_w(self.rails, self.hwmon_path)
                self._samples.append((time.perf_counter() - self._t0, powers))
            except Exception:
                pass  # 순간 read 실패는 무시(다음 샘플로 이어감)
            time.sleep(self.interval_s)

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _sum_series(self, sum_rails) -> list:
        """샘플에서 지정 rail 집합의 합만 뽑아 [(t, power_w), ...]로 반환."""
        return [(t, sum(p.get(r, 0.0) for r in sum_rails)) for t, p in self._samples]

    def mean_power_w(self, sum_rails) -> float:
        series = self._sum_series(sum_rails)
        if not series:
            return 0.0
        return sum(p for _, p in series) / len(series)

    def energy_j(self, sum_rails) -> float:
        """샘플을 사다리꼴로 시간 적분해 총 에너지(J)를 구한다."""
        series = self._sum_series(sum_rails)
        if len(series) < 2:
            return 0.0
        e = 0.0
        for (t0, p0), (t1, p1) in zip(series, series[1:]):
            e += (p0 + p1) / 2.0 * (t1 - t0)
        return e

    def duration_s(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]


def measure_idle_baseline(duration_s: float = 30.0, interval_s: float = 0.1) -> dict:
    """
    모델 로딩 전 idle 상태 평균 전력(BE)을 compute/total 두 가지로 측정한다.
    duration_s 기본값 30초(확정값).
    반환: {"compute": W, "total": W}
    """
    print(f"[energy] idle-baseline 측정 시작 ({duration_s:.0f}s) …")
    sampler = SysfsPowerSampler(interval_s=interval_s)
    with sampler:
        time.sleep(duration_s)
    be = {
        "compute": sampler.mean_power_w(COMPUTE_RAILS),
        "total": sampler.mean_power_w(TOTAL_RAILS),
    }
    print(f"[energy] idle-baseline = compute {be['compute']:.3f} W / total {be['total']:.3f} W")
    return be


def _one_metric_summary(sampler: SysfsPowerSampler, sum_rails, idle_w: float, n_actions: int, suffix: str) -> dict:
    te = sampler.mean_power_w(sum_rails)
    dur = sampler.duration_s()
    net_power = max(te - idle_w, 0.0)  # 음수 방지(노이즈로 BE가 더 클 수 있음)
    e_total = sampler.energy_j(sum_rails)
    e_net = net_power * dur
    mj_per_action = (e_net / max(n_actions, 1)) * 1000.0
    return {
        f"idle_power_w_{suffix}": round(idle_w, 4),
        f"active_power_w_{suffix}": round(te, 4),
        f"net_power_w_{suffix}": round(net_power, 4),
        f"energy_j_total_{suffix}": round(e_total, 4),
        f"energy_j_net_{suffix}": round(e_net, 4),
        f"energy_mj_per_action_{suffix}": round(mj_per_action, 4),
    }


def energy_summary(sampler: SysfsPowerSampler, idle_baseline: dict, n_actions: int) -> dict:
    """
    측정 구간 sampler로부터 compute(1+2)·total(1+2+3) 두 지표의 에너지 요약을 만든다.
    idle_baseline: measure_idle_baseline()이 반환한 {"compute":.., "total":..} dict.
    반환 dict는 두 지표의 컬럼(_compute / _total 접미사)을 모두 포함한다.
    """
    out = {}
    out.update(_one_metric_summary(sampler, COMPUTE_RAILS, idle_baseline["compute"], n_actions, "compute"))
    out.update(_one_metric_summary(sampler, TOTAL_RAILS, idle_baseline["total"], n_actions, "total"))
    return out