"""Dataset-specific mappings, independent of the LeRobot writer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import numpy as np


def get_field(value: Any, path: str) -> Any:
    """Read a dotted path from decoded ROS objects or JSON dictionaries."""
    for part in path.split("."):
        value = value[part] if isinstance(value, dict) else getattr(value, part)
    return value


@dataclass(frozen=True)
class Source:
    topic: str
    field: str
    names: tuple[str, ...]
    mapping: str = "array"
    joints: tuple[str, ...] = ()
    names_field: str = "name"
    indices: tuple[int, ...] = ()
    unit: str | None = None
    max_age_ms: float = 300.0
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.topic or not self.field or not self.names:
            raise ValueError("Each source needs topic, field and nonempty names")
        if self.mapping not in ("array", "by_name", "scalar"):
            raise ValueError(f"Unsupported mapping: {self.mapping}")
        if self.mapping == "by_name" and len(self.joints) != len(self.names):
            raise ValueError("by_name requires one joint per output name")
        if self.mapping == "scalar" and len(self.names) != 1:
            raise ValueError("scalar requires exactly one output name")
        if self.indices and (len(self.indices) != len(self.names)
                             or any(type(i) is not int or i < 0 for i in self.indices)):
            raise ValueError("indices must be nonnegative integers, one per output name")
        if (not np.isfinite([self.max_age_ms, self.scale, self.offset]).all()
                or self.max_age_ms <= 0):
            raise ValueError("Invalid max_age_ms, scale or offset")

    def extract(self, message: Any) -> np.ndarray:
        values = np.asarray(get_field(message, self.field), dtype=np.float64)
        if self.mapping == "scalar":
            if values.ndim != 0:
                raise ValueError(f"{self.field}: expected scalar")
            values = values.reshape(1)
        elif values.ndim != 1:
            raise ValueError(f"{self.field}: expected one-dimensional numeric array")
        if self.mapping == "by_name":
            incoming = list(get_field(message, self.names_field))
            if len(incoming) != len(set(incoming)) or len(incoming) != len(values):
                raise ValueError(f"{self.topic}: duplicate names or array length mismatch")
            lookup = {name: i for i, name in enumerate(incoming)}
            missing = set(self.joints) - lookup.keys()
            if missing:
                raise ValueError(f"{self.topic}: missing required joints {sorted(missing)}")
            values = values[[lookup[name] for name in self.joints]]
        elif self.indices:
            values = values[list(self.indices)]
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(values * self.scale + self.offset, dtype=np.float32)
        if values.shape != (len(self.names),) or not np.isfinite(values).all():
            raise ValueError(f"{self.topic}.{self.field}: expected {len(self.names)} finite values")
        return values


@dataclass(frozen=True)
class Camera:
    topic: str
    info_topic: str | None = None
    metadata_topic: str | None = None


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    state: tuple[Source, ...]
    action: tuple[Source, ...]
    cameras: dict[str, Camera] = field(default_factory=dict)
    robot_type: str = "mcap"
    task: str = "Converted from MCAP"
    action_alignment: str = "previous"
    timestamp_source: str = "log"
    require_capture_metadata: bool = False
    require_configuration: bool = False
    check_action_jumps: bool = False
    check_urdf: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.state or not self.action:
            raise ValueError("Profile needs name, state and action sources")
        if self.action_alignment not in ("previous", "nearest"):
            raise ValueError("action_alignment must be previous or nearest")
        if self.timestamp_source not in ("header", "log"):
            raise ValueError("timestamp_source must be header or log")
        for sources in (self.state, self.action):
            names = [name for source in sources for name in source.names]
            if len(set(names)) != len(names):
                raise ValueError("Output names must be unique within state/action")
        for name in self.cameras:
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
                raise ValueError(f"Invalid camera feature name: {name!r}")

    @classmethod
    def load(cls, path: Path) -> DatasetProfile:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for key in ("state", "action"):
            sources = []
            for item in data[key]:
                item = dict(item)
                for name in ("names", "joints", "indices"):
                    if name in item:
                        item[name] = tuple(item[name])
                sources.append(Source(**item))
            data[key] = tuple(sources)
        data["cameras"] = {name: Camera(**value) for name, value in data.get("cameras", {}).items()}
        return cls(**data)


def read_profile_episode(source: Path, config: Any, profile: DatasetProfile):
    """First pass stores scalar streams and image scores, not RGB arrays."""
    from data_processing_tool import (
        RawEpisode, Stream, decoded_messages, image_quality, issue, message_time, rgb_image,
    )

    source = source.resolve()
    meta_path = source.parent.parent / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    root = source.parent.parent.parent
    snapshots = {n: json.loads((root / f"{n}.json").read_text(encoding="utf-8"))
                 for n in ("sensor", "controller", "robot") if (root / f"{n}.json").exists()}
    raw = RawEpisode(source, metadata, snapshots, profile=profile)
    raw.commands = {name: Stream() for item in profile.action for name in item.names}
    raw.state_channels = {name: Stream() for item in profile.state for name in item.names}
    raw.images = {name: Stream() for name in profile.cameras}
    if profile.require_capture_metadata and (metadata.get("state") != "COMPLETE"
            or not metadata.get("bag_validation", {}).get("valid", False)):
        issue(raw, "error", "capture_rejected", "metadata")
    if profile.require_configuration and not {"sensor", "controller"} <= snapshots.keys():
        issue(raw, "error", "missing_configuration", "sensor/controller")
    import hashlib
    for name, expected in metadata.get("custom_metadata", {}).get("dataset_parameters", {}).get("sha256", {}).items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            issue(raw, "error", "configuration_hash_mismatch", name)
    topics = {s.topic for s in (*profile.state, *profile.action)}
    for camera in profile.cameras.values():
        topics.update(t for t in (camera.topic, camera.info_topic) if t)
    for channel, message, obj in decoded_messages(source, sorted(topics)):
        topic = channel.topic
        raw.counts[topic] = raw.counts.get(topic, 0) + 1
        try:
            timestamp = message_time(message, obj, config)
            # Extract all selected fields before mutating any stream for this message.
            updates = []
            for sources, streams in ((profile.state, raw.state_channels), (profile.action, raw.commands)):
                for spec in sources:
                    if spec.topic == topic:
                        if streams is raw.commands and spec.mapping == "by_name":
                            # Command publishers may update only a subset of joints.
                            # Keep separate clocks; missing joints are checked as empty/stale later.
                            incoming = list(get_field(obj, spec.names_field))
                            values = np.asarray(get_field(obj, spec.field), dtype=np.float64)
                            if (values.ndim != 1 or len(values) != len(incoming)
                                    or len(set(incoming)) != len(incoming)):
                                raise ValueError(f"{topic}: invalid command names/values")
                            selected = [(joint, name) for joint, name in zip(spec.joints, spec.names)
                                        if joint in incoming]
                            if not selected:
                                continue
                            spec = replace(spec, joints=tuple(j for j, _ in selected),
                                           names=tuple(n for _, n in selected))
                        updates.extend((streams[name], float(value))
                                       for name, value in zip(spec.names, spec.extract(obj)))
            for stream, value in updates:
                stream.times.append(timestamp)
                stream.values.append(value)
            for name, camera in profile.cameras.items():
                if topic == camera.topic:
                    rgb = rgb_image(obj)
                    raw.images[name].times.append(timestamp)
                    raw.images[name].values.append({"shape": list(rgb.shape), **image_quality(rgb)})
                if topic == camera.info_topic:
                    record = {"image_size": [int(obj.width), int(obj.height)],
                              "frame_id": obj.header.frame_id, "k": list(obj.k), "d": list(obj.d),
                              "r": list(obj.r), "p": list(obj.p), "distortion_model": obj.distortion_model}
                    if name in raw.calibration and raw.calibration[name] != record:
                        issue(raw, "error", "calibration_changed", name)
                    raw.calibration[name] = record
        except (ValueError, KeyError, AttributeError, TypeError, IndexError) as exc:
            issue(raw, "error", "invalid_message", topic, detail=str(exc),
                  message_index=raw.counts[topic] - 1)
    return raw
