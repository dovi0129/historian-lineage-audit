"""OPC UA server (asyncua) that replays TEP values, one Double variable per channel.

publish(tick, values) sets every channel's Value with SourceTimestamp = process time of the tick
(T0 + tick minutes, the same axis as IoTDB time) and ServerTimestamp = the server's wall clock at the write.
asyncua does not stamp ServerTimestamp on server-side writes, so this module stamps it at the moment the value
enters the address space.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from asyncua import Server, ua

from historian_io import T0_MS

T0_DT = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=T0_MS)
NS_URI = "urn:historian-observed-testbed:tep"


def process_time(tick: int) -> datetime:
    return T0_DT + timedelta(minutes=tick)


class TEPReplayServer:
    def __init__(self, port: int, channels: list[str]):
        self.endpoint = f"opc.tcp://127.0.0.1:{port}/tep/"
        self.channels = channels
        self.server = Server()
        self.nodes: dict = {}
        logging.getLogger("asyncua").setLevel(logging.WARNING)

    async def start(self) -> None:
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name("TEP replay (historian observed testbed)")
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        idx = await self.server.register_namespace(NS_URI)
        obj = await self.server.nodes.objects.add_object(idx, "TEP")
        for ch in self.channels:
            node = await obj.add_variable(ua.NodeId(ch, idx), ch, ua.Variant(0.0, ua.VariantType.Double))
            self.nodes[ch] = node
        await self.server.start()

    async def publish(self, tick: int, values: dict[str, float]) -> None:
        src = process_time(tick)
        for ch in self.channels:
            dv = ua.DataValue(ua.Variant(values[ch], ua.VariantType.Double), StatusCode=ua.StatusCode(ua.StatusCodes.Good),
                              SourceTimestamp=src, ServerTimestamp=datetime.now(timezone.utc))
            await self.server.write_attribute_value(self.nodes[ch].nodeid, dv)

    async def stop(self) -> None:
        await self.server.stop()
