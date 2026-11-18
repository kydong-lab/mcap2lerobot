"""Juliet raw ROS2 MCAP -> official LeRobot v3.0 writer, with preflight QC.

Run: python juliet_mcap_to_lerobot.py --help
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import logging
import re
from pathlib import Path
import sys
import uuid
from typing import Any, Callable

from quality_report import write_episode_status, write_quality_report

from data_processing_tool import (
    PreparedEpisode, QualityConfig, juliet_profile,
    iter_camera_frames, prepare_episode, read_episode,
)
from mcap_profile import DatasetProfile

LOGGER = logging.getLogger(__name__)
DATASET_VERSION = "0.0.1"


def create_dataset(video_codec: str = "h264", **kwargs: Any):
    """Use the official writer across supported LeRobot 0.5/0.6 APIs."""
    import inspect
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parameters = inspect.signature(LeRobotDataset.create).parameters
    if "video_backend" in parameters:
        kwargs["video_backend"] = "pyav"
    if "vcodec" in parameters:
        kwargs["vcodec"] = video_codec
    elif "rgb_encoder" in parameters:
        from lerobot.configs.video import RGBEncoderConfig
        kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec=video_codec)
    else:
        raise RuntimeError("Unsupported LeRobot encoder API; install lerobot>=0.5.1,<0.7")
    kwargs["features"] = copy.deepcopy(kwargs["features"])
    return LeRobotDataset.create(**kwargs)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def discover_episodes(root: Path) -> list[Path]:
    """Reject split ROS bags; standalone files are independent episodes."""
    files = sorted(root.rglob("*.mcap")) if root.is_dir() else [root]
    groups: dict[Path, list[Path]] = {}
    for path in files:
        if path.is_file() and path.suffix == ".mcap" and not path.name.startswith("."):
            groups.setdefault(path.parent, []).append(path)
    if not groups:
        raise ValueError(f"No MCAP files found in {root}")
    for bag, paths in groups.items():
        if len(paths) != 1 and (bag.name == "bag" or (bag / "metadata.yaml").exists()
                                or any(re.search(r"_\d+$", p.stem) for p in paths)):
            raise ValueError(f"Multi-file bag not supported yet: {bag}; do not convert parts as episodes")
    return [path for paths in groups.values() for path in paths]


def episode_name(source: Path, profile: DatasetProfile) -> str:
    return source.parent.parent.name if profile.require_capture_metadata else source.stem


def build_features(prepared: PreparedEpisode) -> dict[str, Any]:
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(prepared.raw.state_names),),
                              "names": prepared.raw.state_names},
        "action": {"dtype": "float32", "shape": (len(prepared.raw.action_names),),
                   "names": prepared.raw.action_names},
    }
    for name in prepared.raw.images:
        features[f"observation.images.{name}"] = {
            "dtype": "video", "shape": tuple(prepared.raw.images[name].values[0]["shape"]),
            "names": ["height", "width", "channels"],
        }
    return features


def load_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    result = {}
    if path:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["source_episode"]
            if key in result:
                raise ValueError(f"Duplicate annotation episode: {key}")
            result[key] = row
    return result


def annotation_row(prepared: PreparedEpisode, episode_index: int, fps: int,
                   supplied: dict[str, dict[str, Any]]) -> dict[str, Any]:
    name = episode_name(prepared.raw.source, prepared.raw.profile)
    row = copy.deepcopy(supplied.get(name, {"instruction_segments": []}))
    grid_mode = prepared.report.get("grid_mode")
    if name in supplied and (row.get("grid_start_ns") != int(prepared.times[0]) or row.get("fps") != fps
                             or row.get("grid_mode", "fixed") != grid_mode):
        raise ValueError(f"{name}: annotation timeline differs; expected grid_start_ns, fps and grid_mode from audit")
    segments = row.get("instruction_segments")
    if not isinstance(segments, list):
        raise ValueError("instruction_segments must be a list")
    for segment in segments:
        start, end = segment["start_frame_index"], segment["end_frame_index"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(prepared.times)):
            raise ValueError(f"{name}: invalid [start,end) annotation interval")
        segment.setdefault("result", None)
        if segment["result"] not in (None, "success", "failure"):
            raise ValueError("result must be success, failure or null")
        success_frame = segment.get("success_frame_index")
        if success_frame is not None and (type(success_frame) is not int or not start <= success_frame < end):
            raise ValueError("success_frame_index must be an actual frame within [start,end), or null")
    row.update(episode_index=episode_index, source_episode=name, fps=fps,
               grid_start_ns=int(prepared.times[0]), grid_mode=grid_mode)
    return row


def write_extensions(root: Path, episodes: list[PreparedEpisode], annotations: list[dict[str, Any]],
                     config: QualityConfig, report: dict[str, Any]) -> None:
    """Store deduplicated device/calibration versions and referenced URDFs."""
    devices, calibrations, urdfs = {}, {}, {}

    def identifier(prefix, value):
        return prefix + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]

    for ep in episodes:
        content = ep.raw.snapshots.get("robot", {}).get("urdf", {}).get("content")
        if content:
            urdfs.setdefault(identifier("urdf_", content), content)
    urdf_paths = {key: "meta/juliet/" + ("robot.urdf" if len(urdfs) == 1 else key + ".urdf")
                  for key in urdfs}
    (root / "meta/juliet").mkdir(parents=True, exist_ok=True)
    for key, content in urdfs.items():
        (root / urdf_paths[key]).write_text(content, encoding="utf-8")
    for index, ep in enumerate(episodes):
        snapshot = ep.raw.snapshots
        content = snapshot.get("robot", {}).get("urdf", {}).get("content")
        device = {
            "cameras": snapshot.get("sensor", {}).get("cameras", {}),
            "sensor": {k: v for k, v in snapshot.get("sensor", {}).items() if k != "cameras"},
            "controller": snapshot.get("controller", {}),
            "robot": {k: v for k, v in snapshot.get("robot", {}).items() if k != "urdf"},
            "urdf_path": urdf_paths.get(identifier("urdf_", content)) if content else None,
        }
        device_id = identifier("devices_", device)
        calib_id = identifier("calib_", ep.raw.calibration)
        devices.setdefault(device_id, device)
        calibrations.setdefault(calib_id, {"cameras": ep.raw.calibration})
        custom = ep.raw.metadata.get("custom_metadata", {})
        ep.report.update({
            "devices_id": device_id, "calib_id": calib_id,
            "operator_id": custom.get("operator_id"), "session": custom.get("session"),
            "capture_state": ep.raw.metadata.get("state"), "task_result": None,
            "grid_start_ns": int(ep.times[0]),
        })
    path = root / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Installed LeRobot writer did not produce v3.0")
    profile = episodes[0].raw.profile
    info["juliet"] = {
        "dataset_version": DATASET_VERSION,
        "schema_version": "1.4",
        "devices": devices, "calibrations": calibrations, "processing": asdict(config),
    }
    if profile.name == "juliet":
        info["juliet"].update(state_gripper_unit="m", action_position_unit="rad")
    write_json(path, info)
    write_episode_status(root / "meta/juliet/episode_status.jsonl", report)
    write_jsonl(root / "meta/juliet/annotations.jsonl", annotations)


def verify_dataset(root: Path, episodes: list[PreparedEpisode], repo_id: str,
                   verify_loader: bool = True) -> None:
    import av
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["codebase_version"] != "v3.0" or info["total_episodes"] != len(episodes):
        raise ValueError("Wrong dataset version/episode count")
    expected = build_features(episodes[0])
    for name in ("observation.state", "action"):
        if (info["features"][name]["shape"] != list(expected[name]["shape"])
                or info["features"][name]["names"] != expected[name]["names"]):
            raise ValueError(f"Wrong feature schema: {name}")
    total = sum(len(ep.times) for ep in episodes)
    if info["total_frames"] != total:
        raise ValueError("Wrong total_frames")
    if sum(pq.ParquetFile(p).metadata.num_rows for p in (root / "data").rglob("*.parquet")) != total:
        raise ValueError("Wrong parquet row count")
    for camera in episodes[0].raw.images:
        paths = sorted((root / "videos" / f"observation.images.{camera}").rglob("*.mp4"))
        frame_count = 0
        for path in paths:
            with av.open(str(path)) as container:
                frame_count += sum(1 for _ in container.decode(video=0))
        if frame_count != total:
            raise ValueError(f"{camera}: decoded {frame_count} video frames, expected {total}")
    if verify_loader:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
        cursor = 0
        for ep in episodes:
            for row in (cursor, cursor + len(ep.times) - 1):
                item = dataset[row]
                if (tuple(item["observation.state"].shape) != (len(ep.raw.state_names),)
                        or tuple(item["action"].shape) != (len(ep.raw.action_names),)):
                    raise ValueError("LeRobot loader vector shape mismatch")
                for camera in ep.raw.images:
                    shape = ep.raw.images[camera].values[0]["shape"]
                    if tuple(item[f"observation.images.{camera}"].shape) != (3, shape[0], shape[1]):
                        raise ValueError("LeRobot loader image shape mismatch")
            cursor += len(ep.times)


def cleanup_empty_image_directories(root: Path) -> None:
    """Remove only empty image staging directories after successful verification."""
    images = root / "images"
    if not images.is_dir() or images.is_symlink():
        return
    directories = sorted(
        (path for path in images.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts), reverse=True,
    )
    for path in [*directories, images]:
        try:
            if not any(path.iterdir()):
                path.rmdir()
        except OSError as exc:
            LOGGER.warning("Could not clean empty image directory %s: %s", path, exc)
    if images.exists():
        LOGGER.warning("Retaining images directory because it contains files or could not be cleaned: %s", images)


def convert(input_root: Path, output_root: Path, config: QualityConfig, *,
            repo_id: str = "juliet/pick_place", audit_only: bool = False,
            report_path: Path | None = None, annotation_path: Path | None = None,
            verify_loader: bool = True, dataset_factory: Callable | None = None,
            profile: DatasetProfile | None = None) -> dict[str, Any]:
    """Preflight all inputs, then write into staging and publish only after verification."""
    profile = profile or juliet_profile()
    output_root = output_root.resolve()
    report_path = report_path or output_root.with_name(output_root.name + ".quality_report.html")
    status_path = output_root.with_name(output_root.name + ".episode_status.jsonl")
    if not audit_only and output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    report = {"config": asdict(config), "profile": asdict(profile), "status": "audited",
              "repo_id": repo_id, "output_root": str(output_root),
              "episode_status_path": str(status_path), "episodes": []}
    accepted = []
    for source in discover_episodes(input_root):
        LOGGER.info("Checking %s", episode_name(source, profile))
        try:
            raw = read_episode(source, config, profile)
            prepared = prepare_episode(raw, config)
            episode_report = prepared.report if isinstance(prepared, PreparedEpisode) else prepared
            custom = raw.metadata.get("custom_metadata", {})
            episode_report.update(operator_id=custom.get("operator_id"), session=custom.get("session"),
                                  capture_state=raw.metadata.get("state"))
        except Exception as exc:
            prepared = {"source": str(source), "source_episode": episode_name(source, profile),
                        "status": "failed", "frames": 0,
                        "issues": [{"severity": "error", "kind": "read_error", "detail": str(exc)}]}
        if isinstance(prepared, PreparedEpisode):
            prepared.report["output_episode_index"] = len(accepted)
            accepted.append(prepared)
            report["episodes"].append(prepared.report)
        else:
            report["episodes"].append(prepared)
    report["accepted_episodes"] = len(accepted)
    report["rejected_episodes"] = len(report["episodes"]) - len(accepted)
    write_quality_report(report_path, report)
    write_episode_status(status_path, report)
    if audit_only:
        return report
    if not accepted:
        raise ValueError(f"No episodes passed QC. See {report_path}")
    supplied = load_annotations(annotation_path)
    accepted_names = {episode_name(ep.raw.source, profile) for ep in accepted}
    if len(accepted_names) != len(accepted):
        raise ValueError("Duplicate episode names; rename source files before converting")
    if set(supplied) - accepted_names:
        raise ValueError("Annotations reference rejected/unknown episodes; nothing was written")
    annotations = [annotation_row(ep, i, config.fps, supplied) for i, ep in enumerate(accepted)]
    features = build_features(accepted[0])
    if any(build_features(ep) != features for ep in accepted[1:]):
        raise ValueError("Image shape/schema changed between episodes; split into separate datasets")
    if dataset_factory is None:
        dataset_factory = create_dataset
    staging = output_root.with_name(f".{output_root.name}.inprogress-{uuid.uuid4().hex[:8]}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = None
    try:
        dataset = dataset_factory(repo_id=repo_id, root=staging, fps=config.fps, robot_type=profile.robot_type,
                                  features=copy.deepcopy(features), use_videos=bool(profile.cameras), video_codec="h264")
        for index, ep in enumerate(accepted):
            LOGGER.info("Writing episode %d (%d frames)", index, len(ep.times))
            generators = {camera: iter_camera_frames(ep, camera) for camera in ep.raw.images}
            task = ep.raw.metadata.get("custom_metadata", {}).get("task", profile.task)
            try:
                for row in range(len(ep.times)):
                    frame = {"observation.state": ep.states[row].copy(), "action": ep.actions[row].copy(),
                             "task": task}
                    frame.update({f"observation.images.{camera}": next(gen) for camera, gen in generators.items()})
                    dataset.add_frame(frame)
                dataset.save_episode()
            finally:
                for gen in generators.values():
                    gen.close()
        dataset.finalize()
        report["status"] = "converted"
        write_extensions(staging, accepted, annotations, config, report)
        verify_dataset(staging, accepted, repo_id, verify_loader)
        report["verification"] = {"schema_counts_video_frames": True, "official_loader": verify_loader}
        cleanup_empty_image_directories(staging)
        staging.rename(output_root)
    except Exception as exc:
        report.update(status="conversion_failed", error=str(exc), staging_path=str(staging))
        write_quality_report(report_path, report)
        write_episode_status(status_path, report)
        raise
    report["episode_status_path"] = str(output_root / "meta/juliet/episode_status.jsonl")
    write_quality_report(report_path, report)
    try:
        status_path.unlink(missing_ok=True)
    except OSError as exc:
        LOGGER.warning("Could not remove provisional episode ledger %s: %s", status_path, exc)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Collection root (scanned recursively)")
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory; must not exist")
    parser.add_argument("--config", type=Path, help="JSON QualityConfig overrides")
    parser.add_argument("--profile", type=Path, help="JSON state/action/camera mappings; default: Juliet")
    parser.add_argument("--timestamp-source", choices=["header", "log"])
    parser.add_argument("--fps", type=int)
    parser.add_argument("--repo-id", default="juliet/pick_place")
    parser.add_argument("--audit-only", action="store_true", help="Run QC without importing LeRobot/Torch")
    parser.add_argument("--report", type=Path, help="HTML quality report path")
    parser.add_argument("--annotations", type=Path, help="Reviewed JSONL with source_episode, grid_start_ns and fps")
    parser.add_argument("--skip-loader-verification", action="store_true",
                        help="Only verify schema/counts, e.g. when video decoder backend is unavailable")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        options = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        profile = DatasetProfile.load(args.profile) if args.profile else juliet_profile()
        options.setdefault("timestamp_source", profile.timestamp_source)
        if args.fps is not None:
            options["fps"] = args.fps
        if args.timestamp_source:
            options["timestamp_source"] = args.timestamp_source
        result = convert(args.input, args.output, QualityConfig(**options), repo_id=args.repo_id,
                         audit_only=args.audit_only, report_path=args.report, annotation_path=args.annotations,
                         verify_loader=not args.skip_loader_verification, profile=profile)
        print(json.dumps({k: v for k, v in result.items() if k not in ("episodes", "config", "profile")}))
        return 0 if result["accepted_episodes"] else 2
    except (ValueError, OSError, ImportError, TypeError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
