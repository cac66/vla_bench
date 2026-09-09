"""
protocol.py
-----------
host(시뮬) ↔ Jetson(모델) 간 관측/행동 직렬화.

- ZMQ REQ/REP 위에서 numpy를 담아 주고받는다.
- 연구용 사설 LAN 전제로 pickle을 쓴다(구현 단순). 신뢰할 수 없는 네트워크에서는 사용 금지.
  운영 환경으로 확장 시 msgpack+검증 스키마로 교체할 것.

메시지 규약
  요청(obs) : {"image": HxWx3 uint8, "image2": ...|None, "state": np.ndarray|None,
               "instruction": str}
  응답(act) : {"action": np.ndarray, "infer_ms": float}
"""

import pickle
import zmq


def send_obj(sock, obj) -> None:
    sock.send(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def recv_obj(sock):
    return pickle.loads(sock.recv())


def make_server_socket(port: int):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{port}")
    return sock


def make_client_socket(host: str, port: int, timeout_ms: int = 10000):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.connect(f"tcp://{host}:{port}")
    return sock
