"""
policy_client.py  (host에서 실행)
---------------------------------
시뮬 harness가 쓰는 원격 정책 래퍼. 관측을 Jetson 서버로 보내고 action을 받는다.
로컬 정책처럼 predict(obs) 한 줄로 쓰도록 만든 drop-in 클래스다.
"""

import time
import numpy as np

from serving.protocol import make_client_socket, send_obj, recv_obj


class RemotePolicy:
    def __init__(self, host: str, port: int = 5555, timeout_ms: int = 15000):
        self.sock = make_client_socket(host, port, timeout_ms)
        self.host, self.port = host, port
        self.net_ms_log = []       # 왕복(네트워크+추론) 시간
        self.infer_ms_log = []     # 서버가 보고한 순수 추론 시간

    def ping(self) -> dict:
        send_obj(self.sock, {"__cmd__": "ping"})
        return recv_obj(self.sock)

    def predict(self, obs: dict) -> np.ndarray:
        """
        obs = {"image": HxWx3 uint8, "image2": ...|None,
               "state": np.ndarray|None, "instruction": str}
        전처리·추론은 서버(Jetson)가 담당한다. 여기선 원시 관측만 보낸다.
        """
        t0 = time.perf_counter()
        send_obj(self.sock, obs)
        resp = recv_obj(self.sock)
        self.net_ms_log.append((time.perf_counter() - t0) * 1000)
        self.infer_ms_log.append(resp.get("infer_ms", float("nan")))
        return np.asarray(resp["action"], dtype=np.float32)

    def stats(self) -> dict:
        def _m(x):
            return float(np.mean(x)) if x else float("nan")
        return {"rtt_ms_mean": _m(self.net_ms_log),
                "server_infer_ms_mean": _m(self.infer_ms_log),
                "n_calls": len(self.net_ms_log)}

    def close(self):
        self.sock.close()
