"""
smoothness.py
-------------
Action smoothness 지표 2종(확정: LDLJ + JerkRMS).
성공한 episode의 action 시퀀스에 대해서만 계산한다(설계서 §3, 실패 궤적 제외).

입력 action_seq: np.ndarray shape [T, action_dim] — closed-loop rollout에서 실제로
                 실행된 action을 시간순으로 쌓은 것.
dt              : step 간 시간 간격(초). 없으면 균일 간격 1.0으로 가정(상대 비교용).
"""

import numpy as np


def _velocity(action_seq: np.ndarray, dt: float) -> np.ndarray:
    """1차 미분(속도) — action 자체를 위치로 취급해 유한차분."""
    return np.diff(action_seq, axis=0) / dt


def _jerk(action_seq: np.ndarray, dt: float) -> np.ndarray:
    """3차 미분(jerk) — position → velocity → accel → jerk, 3번 유한차분."""
    vel = _velocity(action_seq, dt)
    acc = _velocity(vel, dt) if len(vel) > 1 else np.zeros((0, action_seq.shape[1]))
    jerk = _velocity(acc, dt) if len(acc) > 1 else np.zeros((0, action_seq.shape[1]))
    return jerk


def compute_ldlj(action_seq: np.ndarray, dt: float = 1.0) -> float:
    """
    Log Dimensionless Jerk. 값이 클수록(=덜 음수) 부드러움.
    LDLJ = -ln( (T^3 / v_peak^2) * ∫||jerk(t)||^2 dt )
    """
    action_seq = np.asarray(action_seq, dtype=np.float64)
    T = action_seq.shape[0] * dt
    if T <= 0 or action_seq.shape[0] < 5:
        return float("nan")  # 표본이 너무 짧으면 계산 불가

    vel = _velocity(action_seq, dt)
    speed = np.linalg.norm(vel, axis=1)
    v_peak = speed.max() if len(speed) > 0 else 0.0
    if v_peak <= 0:
        return float("nan")

    jerk = _jerk(action_seq, dt)
    jerk_sq_integral = np.sum(np.linalg.norm(jerk, axis=1) ** 2) * dt

    val = (T ** 3 / (v_peak ** 2 + 1e-12)) * (jerk_sq_integral + 1e-12)
    if val <= 0:
        return float("nan")
    return float(-np.log(val))


def compute_jerk_rms(action_seq: np.ndarray, dt: float = 1.0) -> float:
    """
    Jerk RMS. 3차 차분(Δa_t = a_{t+3} - 3a_{t+2} + 3a_{t+1} - a_t)의 RMS.
    값이 작을수록 부드러움.
    """
    a = np.asarray(action_seq, dtype=np.float64)
    L = a.shape[0]
    if L < 4:
        return float("nan")
    diffs = a[3:] - 3 * a[2:-1] + 3 * a[1:-2] - a[:-3]
    diffs = diffs / (dt ** 3)
    sq = np.sum(diffs ** 2, axis=1)  # 차원 결합 노름의 제곱
    rms = float(np.sqrt(np.mean(sq)))
    return rms


def compute_smoothness(action_seq: np.ndarray, dt: float = 1.0, success: bool = True) -> dict:
    """
    성공 episode에만 의미 있는 지표이므로, success=False면 NA를 반환한다(설계서 §3 준수).
    """
    if not success:
        return {"smoothness_ldlj": "NA", "smoothness_jerkrms": "NA"}
    ldlj = compute_ldlj(action_seq, dt)
    jrms = compute_jerk_rms(action_seq, dt)
    return {
        "smoothness_ldlj": "NA" if np.isnan(ldlj) else round(ldlj, 4),
        "smoothness_jerkrms": "NA" if np.isnan(jrms) else round(jrms, 6),
    }
