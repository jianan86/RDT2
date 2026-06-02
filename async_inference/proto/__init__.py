from __future__ import annotations

import sys

from async_inference.proto import rdt2_async_pb2

sys.modules.setdefault("rdt2_async_pb2", rdt2_async_pb2)
