"""Start and stop an isolated, localhost-only Apache IoTDB 1.3.7 standalone instance.

Each instance gets its own conf/, data/ and logs/ under `testbed/instances/<name>/` and its own port block.
The vendor tree (default `testbed/vendor/iotdb/...`, env TESTBED_IOTDB_HOME) is only read (jars and start scripts).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

TESTBED = Path(__file__).resolve().parent
# IoTDB 1.3.7 all-bin tree; override with TESTBED_IOTDB_HOME (distinct from IoTDB's own IOTDB_HOME variable)
IOTDB_HOME = Path(os.environ.get("TESTBED_IOTDB_HOME", TESTBED / "vendor" / "iotdb" / "apache-iotdb-1.3.7-all-bin"))
IOTDB_VERSION = "1.3.7"
INSTANCES = TESTBED / "instances"

DIR_KEYS = {
    "cn_system_dir": "data/confignode/system",
    "cn_consensus_dir": "data/confignode/consensus",
    "cn_pipe_receiver_file_dir": "data/confignode/system/pipe/receiver",
    "dn_system_dir": "data/datanode/system",
    "dn_data_dirs": "data/datanode/data",
    "dn_consensus_dir": "data/datanode/consensus",
    "dn_wal_dirs": "data/datanode/wal",
    "dn_tracing_dir": "datanode/tracing",
    "dn_sync_dir": "data/datanode/sync",
    "sort_tmp_dir": "data/datanode/tmp",
    "dn_pipe_receiver_file_dirs": "data/datanode/system/pipe/receiver",
    "pipe_consensus_receiver_file_dirs": "data/datanode/system/pipe/consensus/receiver",
    "udf_lib_dir": "ext/udf",
    "trigger_lib_dir": "ext/trigger",
    "pipe_lib_dir": "ext/pipe",
    "load_active_listening_dirs": "ext/load/pending",
    "load_active_listening_fail_dir": "ext/load/failed",
}

PORT_KEYS = [
    "cn_internal_port", "cn_consensus_port", "dn_rpc_port", "dn_internal_port", "dn_mpp_data_exchange_port",
    "dn_schema_region_consensus_port", "dn_data_region_consensus_port", "cn_metric_prometheus_reporter_port",
    "dn_metric_prometheus_reporter_port", "rest_service_port", "pipe_air_gap_receiver_port", "mqtt_port",
]

# Settings that matter for when data reach disk. Everything else stays at the 1.3.7 defaults.
FLUSH_SETTINGS = {
    # automatic timed flush off (defaults: true, 600000 ms, checked every 30000 ms)
    "enable_timed_flush_seq_memtable": "false",
    "enable_timed_flush_unseq_memtable": "false",
    "seq_memtable_flush_interval_in_ms": "86400000000",
    "unseq_memtable_flush_interval_in_ms": "86400000000",
    # no background merging of flushed files (defaults: true); keeps the flushed files as written
    "enable_seq_space_compaction": "false",
    "enable_unseq_space_compaction": "false",
    "enable_cross_space_compaction": "false",
    "enable_auto_repair_compaction": "false",
    # one data region so that one FLUSH covers one memtable (default 2)
    "default_data_region_group_num_per_database": "1",
    # documented defaults kept explicit; run timestamps stay inside one partition (see collector.T0_MS)
    "time_partition_interval": "604800000",
    "time_partition_origin": "0",
    "timestamp_precision": "ms",
    "wal_mode": "ASYNC",
    # memory-triggered flush (SystemInfo marks a memtable to flush when write memory use exceeds
    # flush_proportion of the write budget; default 0.4). Raised to 0.75, below reject_proportion 0.8 (default,
    # kept: above it writes are rejected with an error, not flushed). A run writes < 1 MB.
    "flush_proportion": "0.75",
    "reject_proportion": "0.8",
}
# Other known memtable flush triggers: WAL size control (wal_file_size_threshold_in_byte 31457280, default; a run
# writes < 1 MB of WAL). 1.3.7 has no per-series point-count trigger (no avg_series_point_number_threshold key in
# its jars). Whatever the trigger, each memtable flush is logged by TsFileProcessor ("The compression ratio of
# tsfile ..."); run_one.py counts those lines and fails the run unless they equal the files sealed by scheduled FLUSHes.
UNCHANGED_NOTES = {
    "memtable_flush_triggers": "FLUSH statement; timed flush disabled; memory control flush_proportion 0.75; "
                               "WAL size control at default (31457280 B per WAL file); checked by flush log count",
    "localhost_only": "cn_internal_address, dn_rpc_address, dn_internal_address = 127.0.0.1",
}
FLUSH_LOG_MARK = "The compression ratio of tsfile"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


class IoTDBInstance:
    def __init__(self, name: str, slot: int, cn_mem: str = "512M", dn_mem: str = "2G"):
        self.name = name
        self.root = INSTANCES / name
        self.base_port = 21000 + 20 * slot
        self.ports = {k: self.base_port + i for i, k in enumerate(PORT_KEYS)}
        self.cn_mem, self.dn_mem = cn_mem, dn_mem
        self.procs: dict[str, subprocess.Popen] = {}

    @property
    def rpc_port(self) -> int:
        return self.ports["dn_rpc_port"]

    def settings(self) -> dict:
        d = dict(FLUSH_SETTINGS)
        d.update({k: str(v) for k, v in self.ports.items()})
        d.update({"cn_memory": self.cn_mem, "dn_memory": self.dn_mem, "instance_root": str(self.root)})
        d.update({f"note_{k}": v for k, v in UNCHANGED_NOTES.items()})
        return d

    def prepare(self, fresh: bool = True) -> None:
        if fresh and self.root.exists():
            shutil.rmtree(self.root)
        conf = self.root / "conf"
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        conf.mkdir(parents=True, exist_ok=True)
        src = IOTDB_HOME / "conf"
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, conf / f.name)
        logs = str(self.root / "logs")
        for f in ("logback-datanode.xml", "logback-confignode.xml"):
            p = conf / f
            p.write_text(p.read_text().replace("${IOTDB_HOME}/logs", logs).replace("${CONFIGNODE_HOME}/logs", logs))
        for f, mem in (("confignode-env.sh", self.cn_mem), ("datanode-env.sh", self.dn_mem)):
            p = conf / f
            p.write_text(p.read_text().replace("\nMEMORY_SIZE=\n", f"\nMEMORY_SIZE={mem}\n", 1))
        props = {
            "cluster_name": f"tb_{self.name}"[:60],
            "cn_seed_config_node": f"127.0.0.1:{self.ports['cn_internal_port']}",
            "dn_seed_config_node": f"127.0.0.1:{self.ports['cn_internal_port']}",
            "cn_internal_address": "127.0.0.1",
            "dn_rpc_address": "127.0.0.1",
            "dn_internal_address": "127.0.0.1",
            "schema_replication_factor": "1",
            "data_replication_factor": "1",
        }
        props.update({k: str(self.root / v) for k, v in DIR_KEYS.items()})
        props.update({k: str(v) for k, v in self.ports.items()})
        props.update(FLUSH_SETTINGS)
        (conf / "iotdb-system.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))
        (self.root / "instance.json").write_text(json.dumps(self.settings(), indent=1))

    def _env(self) -> dict:
        e = dict(os.environ)
        conf, logs = str(self.root / "conf"), str(self.root / "logs")
        e.update({
            "IOTDB_HOME": str(IOTDB_HOME), "IOTDB_DATA_HOME": str(self.root), "IOTDB_CONF": conf,
            "IOTDB_LOG_DIR": logs, "CONFIGNODE_HOME": str(IOTDB_HOME), "CONFIGNODE_DATA_HOME": str(self.root),
            "CONFIGNODE_CONF": conf, "CONFIGNODE_LOG_DIR": logs, "CONFIGNODE_LOGS": logs,
        })
        return e

    def start(self, timeout: float = 120.0) -> float:
        busy = [p for p in self.ports.values() if not port_free(p)]
        if busy:
            raise RuntimeError(f"ports busy for instance {self.name}: {busy}")
        t0 = time.time()
        env = self._env()
        for node, script in (("confignode", "start-confignode.sh"), ("datanode", "start-datanode.sh")):
            out = open(self.root / "logs" / f"{node}_stdout.log", "ab")
            self.procs[node] = subprocess.Popen(
                ["bash", str(IOTDB_HOME / "sbin" / script), "-f"], env=env, cwd=str(self.root),
                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            if node == "confignode":
                self._wait_port(self.ports["cn_internal_port"], timeout, t0)
        self._wait_port(self.rpc_port, timeout, t0)
        self._wait_session(timeout, t0)
        return time.time() - t0

    def _wait_port(self, port: int, timeout: float, t0: float) -> None:
        while time.time() - t0 < timeout:
            for n, p in self.procs.items():
                if p.poll() is not None:
                    raise RuntimeError(f"{n} exited early with {p.returncode}; see {self.root/'logs'}")
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    return
            time.sleep(0.25)
        raise TimeoutError(f"port {port} not open after {timeout}s")

    def _wait_session(self, timeout: float, t0: float) -> None:
        from iotdb.Session import Session
        last = None
        while time.time() - t0 < timeout:
            try:
                s = Session("127.0.0.1", self.rpc_port, "root", "root")
                s.open(False)
                ds = s.execute_query_statement("show datanodes")
                rows = []
                while ds.has_next():
                    rows.append(str(ds.next()))
                ds.close_operation_handle()
                s.close()
                if any("Running" in r for r in rows):
                    return
                last = rows
            except Exception as exc:  # noqa: BLE001 - retried until timeout
                last = exc
            time.sleep(0.5)
        raise TimeoutError(f"datanode not Running after {timeout}s: {last}")

    def sealed_tsfiles(self) -> int:
        """Number of sealed TsFiles (a .tsfile with its .tsfile.resource) under this instance's data directory."""
        n = 0
        for dirpath, _dirs, files in os.walk(self.root / "data" / "datanode" / "data"):
            names = set(files)
            n += sum(1 for f in files if f.endswith(".tsfile") and f + ".resource" in names)
        return n

    def memtable_flushes_logged(self) -> int:
        log = self.root / "logs" / "log_datanode_all.log"
        with log.open(encoding="utf-8", errors="replace") as fh:
            return sum(FLUSH_LOG_MARK in line for line in fh)

    def stop(self, timeout: float = 60.0) -> None:
        for node in ("datanode", "confignode"):
            p = self.procs.get(node)
            if p is None or p.poll() is not None:
                continue
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        self.procs.clear()
