"""Juliet ROS2 readers, causal resampling and non-destructive quality checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from mcap_profile import Camera, DatasetProfile, Source, read_profile_episode

CAMERAS = {"head": "head", "hand_left": "left", "hand_right": "right"}
ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
                "wrist_yaw", "wrist_pitch", "wrist_roll")
ARMS = [f"{side}_{name}_joint" for side in ("left", "right") for name in ARM_SUFFIXES]
GRIPPERS = ["left_gripper_joint", "right_gripper_joint"]
ACTION_NAMES = GRIPPERS + ARMS + ["waist_yaw_joint"]
BODY = ["neck_pitch_joint", "neck_yaw_joint", "waist_yaw_joint",
        "hip_pitch_joint", "knee_joint", "ankle_pitch_joint"]
STATE_NAMES = ([f"{n}.position" for n in GRIPPERS + ARMS]
               + [f"{n}.velocity" for n in ARMS]
               + [f"{n}.effort" for n in ARMS]
               + [f"{n}.position" for n in BODY])
COMMAND_TOPICS = {"/arms_streaming_controller/command": ARMS,
                  "/juliet_teleop/gripper_command": GRIPPERS,
                  "/waist_streaming_controller/command": ["waist_yaw_joint"]}


def juliet_profile() -> DatasetProfile:
    """The original Juliet schema, expressed through the same mapping interface."""
    state = tuple(Source("/joint_states", field_name,
                         tuple(f"{n}.{field_name}" for n in joints),
                         mapping="by_name", joints=tuple(joints))
                  for field_name, joints in (("position", GRIPPERS + ARMS),
                                             ("velocity", ARMS), ("effort", ARMS),
                                             ("position", BODY)))
    action = tuple(Source(topic, "position", tuple(joints), mapping="by_name",
                          joints=tuple(joints), unit="rad",
                          max_age_ms=30000.0 if joints == GRIPPERS else 150.0)
                   for topic, joints in COMMAND_TOPICS.items())
    # The legacy output orders grippers before arms, independently of topic order.
    action = (action[1], action[0], action[2])
    cameras = {name: Camera(f"/cameras/{side}/color/image_raw",
                           f"/cameras/{side}/color/camera_info",
                           f"/cameras/{side}/color/metadata") for name, side in CAMERAS.items()}
    return DatasetProfile("juliet", state, action, cameras, robot_type="juliet",
                          task="Pick and place.", timestamp_source="header",
                          require_capture_metadata=True, require_configuration=True,
                          check_action_jumps=True, check_urdf=True)


@dataclass(frozen=True)
class QualityConfig:
    fps: int = 30
    timestamp_source: str = "header"
    max_image_gap_ms: float = 15.0
    max_state_gap_ms: float = 5.0
    max_arm_waist_age_ms: float = 150.0
    max_gripper_age_ms: float = 30000.0
    action_jump_rad: float = 0.1
    action_speed_rad_s: float = 2.0
    blur_variance: float = 30.0
    dark_mean: float = 10.0
    bright_mean: float = 245.0
    min_frames: int = 2

    def __post_init__(self) -> None:
        if self.timestamp_source not in ("header", "log"):
            raise ValueError("timestamp_source must be header or log")
        for key, value in asdict(self).items():
            if key != "timestamp_source" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        if not isinstance(self.fps, int) or not isinstance(self.min_frames, int):
            raise ValueError("fps and min_frames must be integers")


@dataclass
class Stream:
    times: list[int] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)


@dataclass
class RawEpisode:
    source: Path
    metadata: dict[str, Any]
    snapshots: dict[str, Any]
    states: Stream = field(default_factory=Stream)
    commands: dict[str, Stream] = field(default_factory=lambda: {n: Stream() for n in ACTION_NAMES})
    images: dict[str, Stream] = field(default_factory=lambda: {n: Stream() for n in CAMERAS})
    calibration: dict[str, Any] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    profile: DatasetProfile = field(default_factory=juliet_profile)
    state_channels: dict[str, Stream] = field(default_factory=dict)

    @property
    def state_names(self) -> list[str]:
        return [name for spec in self.profile.state for name in spec.names]

    @property
    def action_names(self) -> list[str]:
        return [name for spec in self.profile.action for name in spec.names]


@dataclass
class PreparedEpisode:
    raw: RawEpisode
    times: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    image_indices: dict[str, np.ndarray]
    report: dict[str, Any]


def issue(raw: RawEpisode, severity: str, kind: str, field_name: str, **details: Any) -> None:
    raw.issues.append({"severity": severity, "kind": kind, "field": field_name, **details})


def decoded_messages(path: Path, topics: list[str] | None = None) -> Iterator[tuple[Any, Any, Any]]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    with path.open("rb") as stream:
        reader = make_reader(stream)
        factory = DecoderFactory()
        for schema, channel, message in reader.iter_messages(topics=topics, log_time_order=True):
            if channel.message_encoding == "json":
                decoded = json.loads(message.data)
            elif channel.message_encoding == "cdr":
                decoder = factory.decoder_for(channel.message_encoding, schema)
                if decoder is None:
                    raise ValueError(f"No ROS2 decoder for {channel.topic}")
                decoded = decoder(message.data)
            else:
                raise ValueError(f"Unsupported message encoding {channel.message_encoding!r} on {channel.topic}")
            yield channel, message, decoded


def message_time(message: Any, decoded: Any, config: QualityConfig) -> int:
    if config.timestamp_source == "log":
        return int(message.log_time)
    from mcap_profile import get_field
    result = (int(get_field(decoded, "header.stamp.sec")) * 1_000_000_000
              + int(get_field(decoded, "header.stamp.nanosec")))
    if result <= 0:
        raise ValueError("missing/zero header stamp; choose --timestamp-source log explicitly if appropriate")
    return result


def rgb_image(message: Any) -> np.ndarray:
    if message.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    width, height, step = int(message.width), int(message.height), int(message.step)
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if step < width * 3 or data.size != height * step:
        raise ValueError("Image payload/step does not match dimensions")
    rgb = data.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
    return np.ascontiguousarray(rgb if message.encoding == "rgb8" else rgb[:, :, ::-1])


def image_quality(rgb: np.ndarray) -> dict[str, float]:
    # Fixed sampling factor: scores are only comparable with this same algorithm.
    gray = rgb[::2, ::2].astype(np.float32).mean(axis=2)
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
           + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return {"laplacian_variance": float(lap.var()), "mean_brightness": float(gray.mean())}


def state_vector(message: Any) -> np.ndarray:
    """Compatibility helper for the built-in Juliet state mapping."""
    return np.concatenate([spec.extract(message) for spec in juliet_profile().state])


def read_episode(source: Path, config: QualityConfig,
                 profile: DatasetProfile | None = None) -> RawEpisode:
    """First pass keeps vectors, image scores and metadata, never raw image arrays."""
    return read_profile_episode(source, config, profile or juliet_profile())


def nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, target).clip(0, len(source) - 1)
    left = (right - 1).clip(0)
    return np.where(abs(source[right] - target) < abs(source[left] - target), right, left)


def causal_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Never select a future command. Callers must reject the -1 missing prefix."""
    return np.searchsorted(source, target, side="right") - 1


def _timing(raw: RawEpisode, name: str, stream: Stream) -> dict[str, Any]:
    times = np.asarray(stream.times, dtype=np.int64)
    if not len(times):
        issue(raw, "error", "empty_stream", name)
        return {"count": 0}
    diff = np.diff(times)
    if np.any(diff <= 0):
        issue(raw, "error", "non_increasing_timestamp", name,
              indices=np.flatnonzero(diff <= 0).tolist())
    return {"count": len(times), "first_ns": int(times[0]), "last_ns": int(times[-1]),
            "median_hz": float(1e9 / np.median(diff)) if len(diff) and np.median(diff) > 0 else None,
            "max_gap_ms": float(diff.max() / 1e6) if len(diff) else 0.0}


def prepare_episode(raw: RawEpisode, config: QualityConfig) -> PreparedEpisode | dict[str, Any]:
    """Build a CFR grid, then check the exact observations/actions to be written."""
    state_streams = raw.state_channels or {"state": raw.states}
    timing = {f"state/{name}": _timing(raw, name, stream)
              for name, stream in state_streams.items()}
    for name, stream in {**raw.images, **raw.commands}.items():
        timing[name] = _timing(raw, name, stream)
    for camera, stream in raw.images.items():
        if raw.profile.cameras[camera].info_topic and camera not in raw.calibration:
            issue(raw, "error", "missing_camera_info", camera)
        shapes = {tuple(v["shape"]) for v in stream.values}
        calib = raw.calibration.get(camera)
        if len(shapes) != 1 or (calib and shapes and
                              next(iter(shapes))[:2] != tuple(reversed(calib["image_size"]))):
            issue(raw, "error", "image_shape_mismatch", camera)
    source_name = (raw.source.parent.parent.name if raw.profile.require_capture_metadata
                   else raw.source.stem)
    report = {"source": str(raw.source), "source_episode": source_name,
              "capture_state": raw.metadata.get("state"), "counts": raw.counts,
              "timing": timing, "issues": raw.issues, "frames": 0, "cameras": {},
              "completed_checks": ["structure", "timestamps"]}
    def finish() -> dict[str, Any]:
        report["status"] = ("failed" if any(x["severity"] == "error" for x in raw.issues)
                            else "warning" if raw.issues else "passed")
        return report
    if any(x["severity"] == "error" for x in raw.issues):
        return finish()
    streams = [*state_streams.values(), *raw.images.values()]
    start = max([s.times[0] for s in streams] + [s.times[0] for s in raw.commands.values()])
    end = min(s.times[-1] for s in streams)
    usable = raw.metadata.get("usable_start_at_utc")
    if usable:
        start = max(start, int(datetime.fromisoformat(usable).timestamp() * 1e9))
    count = max(0, int((end - start) * config.fps // 1_000_000_000) + 1)
    times = start + np.rint(np.arange(count) * (1e9 / config.fps)).astype(np.int64)
    if count < config.min_frames:
        issue(raw, "error", "insufficient_common_interval", "time", frames=count)
        return finish()
    if raw.state_channels:
        states = np.column_stack([
            np.asarray(raw.state_channels[name].values)[nearest_indices(
                np.asarray(raw.state_channels[name].times, dtype=np.int64), times)]
            for name in raw.state_names]).astype(np.float32)
    else:
        states = np.asarray(raw.states.values, dtype=np.float32)[nearest_indices(
            np.asarray(raw.states.times, dtype=np.int64), times)]
    if states.shape != (count, len(raw.state_names)) or not np.isfinite(states).all():
        issue(raw, "error", "invalid_state_shape", "state")
    image_indices = {}
    for name, stream in {**{f"state/{n}": s for n, s in state_streams.items()}, **raw.images}.items():
        source = np.asarray(stream.times, dtype=np.int64)
        indices = nearest_indices(source, times)
        gaps = abs(source[indices] - times) / 1e6
        is_state = name.startswith("state/")
        limit = config.max_state_gap_ms if is_state else config.max_image_gap_ms
        if np.any(gaps > limit):
            issue(raw, "error", "alignment_gap", name, frames=np.flatnonzero(gaps > limit).tolist(),
                  max_ms=float(gaps.max()), threshold_ms=limit)
        if not is_state:
            image_indices[name] = indices
            scores = [stream.values[i] for i in indices]
            bad = [i for i, s in enumerate(scores) if s["laplacian_variance"] < config.blur_variance
                   or not config.dark_mean <= s["mean_brightness"] <= config.bright_mean]
            if bad:
                issue(raw, "warning", "image_quality", name, frames=bad,
                      blur_threshold=config.blur_variance, dark_threshold=config.dark_mean,
                      bright_threshold=config.bright_mean)
            repeats = np.flatnonzero(np.diff(indices) == 0) + 1
            report["cameras"][name] = {
                "source_frames": len(stream.times), "output_frames": count,
                "reused_frames": len(repeats), "max_alignment_gap_ms": float(gaps.max()),
                "quality_warning_frames": len(bad),
            }
            if len(repeats):
                issue(raw, "warning", "reused_source_image", name, frames=repeats.tolist())
    actions = np.empty((count, len(raw.action_names)), dtype=np.float32)
    age_limits = {name: spec.max_age_ms for spec in raw.profile.action for name in spec.names}
    for column, name in enumerate(raw.action_names):
        stream = raw.commands[name]
        source = np.asarray(stream.times, dtype=np.int64)
        indices = (causal_indices(source, times) if raw.profile.action_alignment == "previous"
                   else nearest_indices(source, times))
        if np.any(indices < 0):
            raise ValueError("internal error: grid starts before first command")
        ages = abs(times - source[indices]) / 1e6
        limit = age_limits[name]
        if raw.profile.name == "juliet":
            limit = config.max_gripper_age_ms if name in GRIPPERS else config.max_arm_waist_age_ms
        if np.any(ages > limit):
            issue(raw, "error", "stale_command", name, frames=np.flatnonzero(ages > limit).tolist(),
                  max_age_ms=float(ages.max()), threshold_ms=limit)
        actions[:, column] = np.asarray(stream.values)[indices]
    delta = abs(np.diff(actions, axis=0))
    speed = delta / (np.diff(times)[:, None] / 1e9)
    bad_rows, bad_columns = np.where((delta > config.action_jump_rad) | (speed > config.action_speed_rad_s))
    if not np.isfinite(actions).all():
        issue(raw, "error", "nonfinite_action", "action")
    for col in np.unique(bad_columns) if raw.profile.check_action_jumps else []:
        rows = bad_rows[bad_columns == col]
        issue(raw, "warning", "action_jump", raw.action_names[col], frames=(rows + 1).tolist(),
              max_delta_rad=float(delta[rows, col].max()), max_speed_rad_s=float(speed[rows, col].max()),
              delta_threshold=config.action_jump_rad, speed_threshold=config.action_speed_rad_s)
    report.update(frames=count, grid_start_ns=int(start), grid_end_ns=int(times[-1]),
                  timestamp_source=config.timestamp_source)
    report["completed_checks"].extend(["alignment", "action_freshness"])
    if raw.images:
        report["completed_checks"].extend(["image_quality", "image_reuse"])
    if raw.profile.check_action_jumps:
        report["completed_checks"].append("action_jumps")
    if raw.profile.check_urdf:
        check_urdf_limits(raw, states, actions)
        if raw.snapshots.get("robot", {}).get("urdf", {}).get("content"):
            report["completed_checks"].append("action_bounds")
    finish()
    if report["status"] == "failed":
        return report
    return PreparedEpisode(raw, times, states, actions, image_indices, report)


def check_urdf_limits(raw: RawEpisode, states: np.ndarray, actions: np.ndarray) -> None:
    """URDF fixed grippers are not physical opening limits; leave them alone."""
    from xml.etree import ElementTree
    content = raw.snapshots.get("robot", {}).get("urdf", {}).get("content")
    if not content:
        issue(raw, "warning", "missing_urdf", "robot")
        return
    root = ElementTree.fromstring(content)
    state_columns = {name.removesuffix(".position"): i for i, name in enumerate(raw.state_names)
                     if name.endswith(".position")}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        name = joint.get("name")
        if joint.get("type") != "revolute" or limit is None or name not in state_columns:
            continue
        lower, upper = float(limit.get("lower", "-inf")), float(limit.get("upper", "inf"))
        values = states[:, state_columns[name]]
        for label, data in [("state", values), ("action", actions[:, raw.action_names.index(name)]
                                               if name in raw.action_names else None)]:
            if data is not None:
                bad = np.flatnonzero((data < lower - 1e-3) | (data > upper + 1e-3))
                if len(bad):
                    issue(raw, "warning", "urdf_limit", f"{label}/{name}", frames=bad.tolist(),
                          lower=lower, upper=upper)


def iter_camera_frames(prepared: PreparedEpisode, camera: str) -> Iterator[np.ndarray]:
    """Second pass yields only selected frames, retaining at most one RGB image."""
    topic = prepared.raw.profile.cameras[camera].topic
    targets = prepared.image_indices[camera]
    cursor = 0
    for ordinal, (_, _, obj) in enumerate(decoded_messages(prepared.raw.source, [topic])):
        if cursor >= len(targets):
            break
        if ordinal != targets[cursor]:
            continue
        rgb = rgb_image(obj)
        while cursor < len(targets) and targets[cursor] == ordinal:
            yield rgb
            cursor += 1
    if cursor != len(targets):
        raise ValueError(f"{camera}: emitted {cursor}/{len(targets)} frames")
