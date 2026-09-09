"""
energy.py
---------
sysfs(INA3221 hwmon) 직접 폴링 기반 에너지 측정.

확정된 하드웨어 매핑
  hwmon 경로 : /sys/class/hwmon/hwmon1
  in1 = VDD_IN            (모듈 전체 전력 레일 — 우리가 쓸 주 지표)
  in2 = VDD_CPU_GPU_CV
  in3 = VDD_SOC
  power 노드 없음 → in*_input(mV) * curr*_input(mA) / 1e6 = W 로 직접 계산.

측정 방법론(설계서 §1 반영)
  - idle-baseline(BE) 측정 → active(TE) 측정 → net = TE - BE
  - 총 에너지(BE 미차감)와 순수 에너지(BE 차감) 둘 다 산출
  - 정규화: energy_mj_per_action (기본), energy_per_gflop(선택, 외부에서 FLOPs 곱)
"""

import os
import time
import threading

HWMON_PATH = "/sys/class/hwmon/hwmon1"
RAIL_LABELS = {1: "VDD_IN", 2: "VDD_CPU_GPU_CV", 3: "VDD_SOC"}
PRIMARY_RAIL = 1  # VDD_IN: 모듈 전체 소비. artifact 비교에는 이 레일을 기본으로 쓴다.


def _read_int(path: str) -> int:
    with open(path, "r") as f:
        return int(f.read().strip())


def read_rail_power_w(rail: int = PRIMARY_RAIL) -> float:
    """
    지정 레일의 순간 전력(W)을 1회 읽는다(open/close 포함, 일회성 점검용).
    power 노드가 없으므로 전압*전류로 계산한다.
    in{rail}_input : mV, curr{rail}_input : mA  →  W = mV*mA/1e6
    연속 폴링(측정 루프)은 SysfsPowerSampler가 fd를 캐싱해 더 빠르게 처리한다.
    """
    v_mv = _read_int(f"{HWMON_PATH}/in{rail}_input")
    i_ma = _read_int(f"{HWMON_PATH}/curr{rail}_input")
    return (v_mv * i_ma) / 1e6


class SysfsPowerSampler:
    """
    백그라운드 스레드로 지정 간격마다 전력을 폴링해 리스트에 적재한다.
    with 블록으로 감싸 사용: 진입 시 폴링 시작, 종료 시 폴링 중지.

    파일을 매 샘플마다 open/close하지 않고 fd를 폴링 구간 동안 열어둔 채
    os.pread로 읽는다 — 100Hz 폴링이 measure 구간(latency 측정)과 동시에
    돌기 때문에, 불필요한 syscall을 줄여 측정 왜곡을 최소화한다.
    """
    def __init__(self, interval_s: float = 0.01, rail: int = PRIMARY_RAIL):
        self.interval_s = interval_s
        self.rail = rail
        self._samples = []          # [(t_rel_s, power_w), ...]
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._v_fd = None
        self._i_fd = None

    def _read_power_w(self) -> float:
        v_mv = int(os.pread(self._v_fd, 32, 0).strip())
        i_ma = int(os.pread(self._i_fd, 32, 0).strip())
        return (v_mv * i_ma) / 1e6

    def _loop(self):
        while not self._stop.is_set():
            try:
                p = self._read_power_w()
                self._samples.append((time.perf_counter() - self._t0, p))
            except Exception:
                pass  # 순간 read 실패는 무시(다음 샘플로 이어감)
            time.sleep(self.interval_s)

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._v_fd = os.open(f"{HWMON_PATH}/in{self.rail}_input", os.O_RDONLY)
        self._i_fd = os.open(f"{HWMON_PATH}/curr{self.rail}_input", os.O_RDONLY)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        for fd in (self._v_fd, self._i_fd):
            if fd is not None:
                os.close(fd)
        self._v_fd = self._i_fd = None

    def mean_power_w(self) -> float:
        if not self._samples:
            return 0.0
        return sum(p for _, p in self._samples) / len(self._samples)

    def energy_j(self) -> float:
        """샘플을 사다리꼴로 시간 적분해 총 에너지(J)를 구한다."""
        if len(self._samples) < 2:
            return 0.0
        e = 0.0
        for (t0, p0), (t1, p1) in zip(self._samples, self._samples[1:]):
            e += (p0 + p1) / 2.0 * (t1 - t0)
        return e

    def duration_s(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]


def measure_idle_baseline(duration_s: float = 30.0, interval_s: float = 0.1,
                          rail: int = PRIMARY_RAIL) -> float:
    """
    모델 로딩 전 idle 상태 평균 전력(BE, W)을 측정한다.
    duration_s 기본값 30초(설계서 확정값).
    """
    print(f"[energy] idle-baseline 측정 시작 ({duration_s:.0f}s) …")
    sampler = SysfsPowerSampler(interval_s=interval_s, rail=rail)
    with sampler:
        time.sleep(duration_s)
    be = sampler.mean_power_w()
    print(f"[energy] idle-baseline = {be:.3f} W")
    return be


def energy_summary(sampler: SysfsPowerSampler, idle_baseline_w: float, n_actions: int) -> dict:
    """
    측정 구간 sampler로부터 에너지 요약 dict를 만든다.
    - active_power_w   : 측정 구간 평균 전력(TE)
    - net_power_w      : TE - BE
    - energy_j_total   : BE 미차감 총 에너지
    - energy_j_net     : BE 차감 순수 에너지 (net_power * duration)
    - energy_mj_per_action : 순수 에너지를 action(=measure_iters) 수로 정규화, mJ 단위
    """
    te = sampler.mean_power_w()
    dur = sampler.duration_s()
    net_power = max(te - idle_baseline_w, 0.0)  # 음수 방지(노이즈로 BE가 더 클 수 있음)
    e_total = sampler.energy_j()
    e_net = net_power * dur
    mj_per_action = (e_net / max(n_actions, 1)) * 1000.0
    return {
        "idle_power_w": round(idle_baseline_w, 4),
        "active_power_w": round(te, 4),
        "net_power_w": round(net_power, 4),
        "energy_j_total": round(e_total, 4),
        "energy_j_net": round(e_net, 4),
        "energy_mj_per_action": round(mj_per_action, 4),
    }
