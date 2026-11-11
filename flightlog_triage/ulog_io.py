"""Read ULog files with pyulog; write small ULog files for tests and demos.

The writer follows the public ULog file format spec (header, flag bits, info, parameter,
format, subscription and data messages). It only supports what the synthetic data needs:
scalar and fixed-size array fields of the basic numeric types.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from .flight import FlightData

# Topics the detectors read. Loading only these keeps parsing fast.
TOPICS = [
    "sensor_combined",
    "vehicle_land_detected",
    "vehicle_status",
    "battery_status",
    "vehicle_gps_position",
    "sensor_gps",
    "estimator_status",
    "input_rc",
    "failsafe_flags",
    "actuator_motors",
    "actuator_outputs",
    "cpuload",
]

INFO_KEYS = ("sys_name", "ver_hw", "ver_sw_release", "sys_mcu")

_SENSOR_KEEP = (
    "t",
    "accelerometer_m_s2[0]",
    "accelerometer_m_s2[1]",
    "accelerometer_m_s2[2]",
    "accelerometer_clipping",
)


def load_ulog(path: str | Path) -> FlightData:
    """Parse a .ulg file into FlightData (only the topics in TOPICS)."""
    from pyulog import ULog  # imported lazily so the package imports without it

    path = Path(path)
    ulog = ULog(str(path), message_name_filter_list=TOPICS, disable_str_exceptions=True)
    t0 = ulog.start_timestamp
    by_name: dict[str, list] = {}
    for d in ulog.data_list:
        by_name.setdefault(d.name, []).append(d)

    topics: dict[str, dict[str, np.ndarray]] = {}
    for name, items in by_name.items():
        chosen = _choose_instance(name, items)
        data = chosen.data
        ts = data["timestamp"].astype(np.float64)
        topic = {"t": (ts - t0) / 1e6}
        for key, arr in data.items():
            if key == "timestamp":
                continue
            topic[key] = np.asarray(arr)
        if name == "sensor_combined":
            topic = {k: v for k, v in topic.items() if k in _SENSOR_KEEP}
        topics[name] = topic

    if "vehicle_gps_position" not in topics and "sensor_gps" in topics:
        topics["vehicle_gps_position"] = topics["sensor_gps"]
    topics.pop("sensor_gps", None)

    info = {k: str(ulog.msg_info_dict[k]) for k in INFO_KEYS if k in ulog.msg_info_dict}
    params = {k: float(v) for k, v in ulog.initial_parameters.items()}
    return FlightData(
        topics=topics,
        params=params,
        info=info,
        source=str(path),
        size_bytes=os.path.getsize(path),
    )


def _choose_instance(name: str, items: list):
    items = sorted(items, key=lambda d: d.multi_id)
    if name == "battery_status":
        # pick the instance that actually measures a pack
        def med_v(d):
            v = d.data.get("voltage_v")
            return float(np.nanmedian(v)) if v is not None and len(v) else 0.0

        return max(items, key=med_v)
    return items[0]


# ----------------------------------------------------------------------------- writer

_TYPES = {
    "int8_t": "i1",
    "uint8_t": "u1",
    "int16_t": "<i2",
    "uint16_t": "<u2",
    "int32_t": "<i4",
    "uint32_t": "<u4",
    "int64_t": "<i8",
    "uint64_t": "<u8",
    "float": "<f4",
    "double": "<f8",
    "bool": "u1",
}

HEADER_MAGIC = b"ULog\x01\x12\x35"
START_US = 1_000_000

# schema: list of (type, name, array_len) with array_len 0 for scalars
Schema = list[tuple[str, str, int]]


def flat_names(schema: Schema) -> list[str]:
    out = []
    for _typ, name, n in schema:
        out.extend([f"{name}[{i}]" for i in range(n)] if n else [name])
    return out


def _msg(msg_type: str, payload: bytes) -> bytes:
    return struct.pack("<HB", len(payload), ord(msg_type)) + payload


def _info(key: str, value: str) -> bytes:
    raw = value.encode()
    k = f"char[{len(raw)}] {key}".encode()
    return _msg("I", struct.pack("<B", len(k)) + k + raw)


def _param(name: str, value: float | int) -> bytes:
    if isinstance(value, int):
        k = f"int32_t {name}".encode()
        return _msg("P", struct.pack("<B", len(k)) + k + struct.pack("<i", value))
    k = f"float {name}".encode()
    return _msg("P", struct.pack("<B", len(k)) + k + struct.pack("<f", value))


def write_ulog(
    path: str | Path,
    flight: FlightData,
    schemas: dict[str, Schema],
    params: dict[str, float | int] | None = None,
) -> Path:
    """Write the topics of ``flight`` that have a schema to a ULog file."""
    path = Path(path)
    out = bytearray()
    out += HEADER_MAGIC + bytes([1]) + struct.pack("<Q", START_US)
    out += _msg("B", bytes(16) + struct.pack("<QQQ", 0, 0, 0))
    for key, value in sorted(flight.info.items()):
        out += _info(key, value)
    for name, value in sorted((params or {}).items()):
        out += _param(name, value)

    names = [n for n in schemas if n in flight.topics]
    for name in names:
        fields = "".join(
            f"{typ}[{n}] {fname};" if n else f"{typ} {fname};" for typ, fname, n in schemas[name]
        )
        out += _msg("F", f"{name}:uint64_t timestamp;{fields}".encode())
    for msg_id, name in enumerate(names):
        out += _msg("A", struct.pack("<BH", 0, msg_id) + name.encode())

    stamps, blobs = [], []
    for msg_id, name in enumerate(names):
        topic = flight.topics[name]
        schema = schemas[name]
        dt = [("size", "<u2"), ("type", "u1"), ("msg_id", "<u2"), ("timestamp", "<u8")]
        cols = []
        for typ, fname, n in schema:
            for flat in [f"{fname}[{i}]" for i in range(n)] if n else [fname]:
                dt.append((flat, _TYPES[typ]))
                cols.append(flat)
        rec = np.zeros(len(topic["t"]), dtype=np.dtype(dt))
        rec["size"] = rec.dtype.itemsize - 3
        rec["type"] = ord("D")
        rec["msg_id"] = msg_id
        ts = np.round(START_US + np.asarray(topic["t"], dtype=float) * 1e6).astype(np.uint64)
        rec["timestamp"] = ts
        for flat in cols:
            if flat in topic:
                rec[flat] = topic[flat]
        raw = rec.tobytes()
        size = rec.dtype.itemsize
        stamps.append(ts)
        blobs.append([raw[i * size : (i + 1) * size] for i in range(len(rec))])

    if stamps:
        all_ts = np.concatenate(stamps)
        owner = np.concatenate([np.full(len(s), k) for k, s in enumerate(stamps)])
        local = np.concatenate([np.arange(len(s)) for s in stamps])
        order = np.argsort(all_ts, kind="stable")
        out += b"".join(blobs[owner[i]][local[i]] for i in order)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


# ----------------------------------------------------------------------------- compact cache


def save_npz(flight: FlightData, path: str | Path) -> Path:
    """Store only the parsed topics (a few % of the raw log size) for re-runs."""
    import json

    arrays = {f"{name}/{key}": arr for name, topic in flight.topics.items() for key, arr in topic.items()}
    meta = json.dumps({"params": flight.params, "info": flight.info, "source": flight.source,
                       "size_bytes": flight.size_bytes})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez_compressed(fh, __meta__=np.array(meta), **arrays)
    return path


def load_npz(path: str | Path) -> FlightData:
    import json

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        topics: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            if key == "__meta__":
                continue
            name, field = key.split("/", 1)
            topics.setdefault(name, {})[field] = z[key]
    return FlightData(topics=topics, params=meta["params"], info=meta["info"], source=meta["source"],
                      size_bytes=int(meta["size_bytes"]))
