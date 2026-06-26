#!/usr/bin/env python3
"""Unit tests for convert_phase1_5_artifacts.py."""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from convert_phase1_5_artifacts import (
    UINT64_MAX,
    get_strategy_params,
    load_params,
    log_skipped,
    make_artifact_dir,
    convert_hybrid,
)
from build_reliability_artifacts import verify_artifact

# ── Helpers ──────────────────────────────────────────────────────

SAMPLE_PARAMS: dict[str, Any] = {
    "metadata": {
        "strategy_count": 3,
        "strategy_ids": [
            "native_scale1",
            "per_dataset",
            "hybrid_scale100_plus_shifted",
        ],
    },
    "strategies": {
        "native_scale1": {
            "per_dataset": {
                "sensor": {"scale": 1, "shift_k": 0},
                "uniform": {"scale": 1, "shift_k": 0},
                "heavy_tailed": {"scale": 1, "shift_k": 0},
                "zipfian": {"scale": 1, "shift_k": 0},
            }
        },
        "per_dataset": {
            "per_dataset": {
                "sensor": {"scale": 8233095970213, "shift_k": 0},
                "uniform": {"scale": 281474977817, "shift_k": 0},
                "heavy_tailed": {"scale": 208, "shift_k": 0},
                "zipfian": {"scale": 47978, "shift_k": 0},
            }
        },
        "hybrid_scale100_plus_shifted": {
            "per_dataset": {
                "sensor": {"scale": 100, "shift_k": 51},
                "uniform": {"scale": 100, "shift_k": 46},
                "heavy_tailed": {"scale": 100, "shift_k": 16},
                "zipfian": {"scale": 100, "shift_k": 23},
            }
        },
    },
}


def uint64_to_planes(x: int) -> list[int]:
    return [(x >> (8 * (7 - p))) & 0xFF for p in range(8)]


def planes_to_uint64(planes: list[int]) -> int:
    return sum(p << (8 * (7 - i)) for i, p in enumerate(planes))


def make_fake_raw(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = struct.pack(f"<{len(values)}d", *values)
    path.write_bytes(buf)


# ── Tests: Byte split / reconstruction ────────────────────────────

def test_uint64_to_planes_roundtrip() -> None:
    cases = [0, 1, 255, 256, 65535, 1 << 56, UINT64_MAX]
    for v in cases:
        planes = uint64_to_planes(v)
        reconstructed = planes_to_uint64(planes)
        assert reconstructed == v, f"roundtrip failed for {v}: got {reconstructed}"


def test_uint64_to_planes_shifted_values() -> None:
    """Verify byte-split of hybrid shifted values."""
    # If we encode raw=15.0 at scale=100: u=1500, then shift_k=51
    # u_shifted = 1500 << 51 = 3377699720527872000
    u = 1500
    shift_k = 51
    u_shifted = u << shift_k
    assert u_shifted == 3377699720527872000

    planes = uint64_to_planes(u_shifted)
    # 3377699720527872000 in hex = 0x2EE0000000000000
    # plane_0 (MSB) = 0x2E = 46, plane_1 = 0xE0 = 224, rest zeros
    assert planes[0] == 0x2E, f"plane_0: expected 0x2E, got {planes[0]}"
    assert planes[1] == 0xE0, f"plane_1: expected 0xE0, got {planes[1]}"
    assert all(p == 0 for p in planes[2:]), f"planes 2-7 should be zero: {planes}"

    reconstructed = planes_to_uint64(planes)
    assert reconstructed == u_shifted, (
        f"reconstructed {reconstructed} != u_shifted {u_shifted}"
    )


# ── Tests: Params loading ─────────────────────────────────────────

def test_load_params() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_PARAMS, f)
        f.flush()
        params_path = Path(f.name)
    try:
        data = load_params(params_path)
        assert "strategies" in data
        assert "native_scale1" in data["strategies"]
        assert "per_dataset" in data["strategies"]
        assert "hybrid_scale100_plus_shifted" in data["strategies"]
    finally:
        params_path.unlink()


def test_load_params_missing_strategy() -> None:
    bad = {"strategies": {"native_scale1": {"per_dataset": {"sensor": {}}}}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        f.flush()
        params_path = Path(f.name)
    try:
        raised = False
        try:
            load_params(params_path)
        except ValueError:
            raised = True
        assert raised, "expected ValueError for missing strategy"
    finally:
        params_path.unlink()


def test_load_params_missing_dataset() -> None:
    bad = {
        "strategies": {
            "native_scale1": {"per_dataset": {"sensor": {"scale": 1}}},
            "per_dataset": {"per_dataset": {"sensor": {"scale": 1}}},
            "hybrid_scale100_plus_shifted": {
                "per_dataset": {"sensor": {"scale": 100, "shift_k": 0}}
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        f.flush()
        params_path = Path(f.name)
    try:
        raised = False
        try:
            load_params(params_path)
        except ValueError:
            raised = True
        assert raised, "expected ValueError for missing dataset"
    finally:
        params_path.unlink()


# ── Tests: Strategy params lookup ─────────────────────────────────

def test_get_strategy_params() -> None:
    scale, shift_k = get_strategy_params(SAMPLE_PARAMS, "native_scale1", "sensor")
    assert scale == 1
    assert shift_k == 0

    scale, shift_k = get_strategy_params(SAMPLE_PARAMS, "per_dataset", "sensor")
    assert scale == 8233095970213
    assert shift_k == 0

    scale, shift_k = get_strategy_params(
        SAMPLE_PARAMS, "hybrid_scale100_plus_shifted", "sensor"
    )
    assert scale == 100
    assert shift_k == 51


# ── Tests: Artifact path construction ─────────────────────────────

def test_make_artifact_dir_native() -> None:
    path = make_artifact_dir(
        Path("/root"), "native_scale1", "sensor", 100000000, 1, 0,
    )
    expected = Path(
        "/root/artifacts_phase1_5/native_scale1/sensor/n100000000/scale1"
    )
    assert path == expected, f"got {path}"


def test_make_artifact_dir_per_dataset() -> None:
    path = make_artifact_dir(
        Path("/root"), "per_dataset", "uniform", 100000000, 281474977817, 0,
    )
    expected = Path(
        "/root/artifacts_phase1_5/per_dataset/uniform/n100000000/"
        "scale281474977817"
    )
    assert path == expected, f"got {path}"


def test_make_artifact_dir_hybrid() -> None:
    path = make_artifact_dir(
        Path("/root"), "hybrid_scale100_plus_shifted", "sensor",
        100000000, 100, 51,
    )
    expected = Path(
        "/root/artifacts_phase1_5/hybrid/sensor/n100000000/scale100_shift51"
    )
    assert path == expected, f"got {path}"


# ── Tests: Hybrid conversion ──────────────────────────────────────

def test_convert_hybrid_tiny() -> None:
    """Convert 5 very small raw FP64 values through the hybrid converter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "test.f64le.bin"
        art_dir = Path(tmpdir) / "artifacts"
        n_rows = 5
        shift_k = 4  # small shift for easy verification

        # Raw values that encode cleanly at scale=100
        raw_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        make_fake_raw(raw_path, raw_values)

        meta = convert_hybrid(
            raw_path, "test", n_rows, shift_k, art_dir, 20260413,
        )

        # Check metadata fields
        assert meta["dataset"] == "test"
        assert meta["n_rows"] == n_rows
        assert meta["scale"] == 100
        assert meta["shift_k"] == shift_k
        assert meta["overflow_count"] == 0

        # Expected: u = round(raw * 100)
        # u_shifted = u << shift_k
        expected_encoded = [int(round(v * 100)) << shift_k for v in raw_values]
        expected_sum = sum(expected_encoded)
        assert int(meta["clean_encoded_sum"]) == expected_sum, (
            f"sum mismatch: {int(meta['clean_encoded_sum'])} != {expected_sum}"
        )

        # Verify plane files exist
        for p in range(8):
            assert (art_dir / f"plane_{p}.bin").exists(), (
                f"missing plane_{p}.bin"
            )
            stat = (art_dir / f"plane_{p}.bin").stat()
            assert stat.st_size == n_rows, (
                f"plane_{p} size {stat.st_size} != {n_rows}"
            )

        # Verify reconstruction matches
        cpu_sum = verify_artifact(art_dir, meta)
        assert cpu_sum == expected_sum, (
            f"reconstruction {cpu_sum} != expected {expected_sum}"
        )

        # Verify artifact.json
        meta_path = art_dir / "artifact.json"
        assert meta_path.exists()
        loaded = json.loads(meta_path.read_text())
        assert loaded["clean_encoded_sum"] == str(expected_sum)
        assert "plane_checksums" in loaded
        assert len(loaded["plane_checksums"]) == 8


def test_convert_hybrid_full_roundtrip() -> None:
    """Convert 10 values, verify every reconstructed value is exact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "test.f64le.bin"
        art_dir = Path(tmpdir) / "artifacts"
        n_rows = 10
        shift_k = 2

        raw_values = [float(i) for i in range(n_rows)]
        make_fake_raw(raw_path, raw_values)

        meta = convert_hybrid(
            raw_path, "test", n_rows, shift_k, art_dir, 20260413,
        )

        expected_encoded = [i * 100 << shift_k for i in range(n_rows)]
        expected_sum = sum(expected_encoded)

        # Check each reconstructed value
        plane_data = []
        for p in range(8):
            data = (art_dir / f"plane_{p}.bin").read_bytes()
            assert len(data) == n_rows
            plane_data.append(data)

        for i in range(n_rows):
            reconstructed = sum(
                plane_data[p][i] << (8 * (7 - p)) for p in range(8)
            )
            assert reconstructed == expected_encoded[i], (
                f"row {i}: reconstructed {reconstructed} "
                f"!= expected {expected_encoded[i]}"
            )

        assert int(meta["clean_encoded_sum"]) == expected_sum


def test_convert_hybrid_overflow_skip() -> None:
    """Shift that overflows uint64 should raise RuntimeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "overflow.f64le.bin"
        art_dir = Path(tmpdir) / "artifacts"
        n_rows = 1
        shift_k = 63  # massive shift: any non-zero u will overflow

        # A raw value that gives u=100, then 100 << 63 > uint64_max
        make_fake_raw(raw_path, [1.0])

        raised = False
        try:
            convert_hybrid(
                raw_path, "overflow_test", n_rows, shift_k,
                art_dir, 20260413,
            )
        except RuntimeError as e:
            if "OVERFLOW" in str(e):
                raised = True
        assert raised, "expected RuntimeError for overflow"


# ── Tests: Skipped CSV logging ────────────────────────────────────

def test_log_skipped_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "skipped.csv"
        entry = {
            "strategy_id": "native_scale1",
            "strategy_scale": 1,
            "strategy_shift_k": 0,
            "dataset": "sensor",
            "n_rows": 1000,
            "reason": "inapplicable_overflow",
            "details": "overflow at row 0",
        }
        log_skipped(csv_path, entry)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "strategy_id" in content
        assert "native_scale1" in content
        assert "sensor" in content


def test_log_skipped_append() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "skipped.csv"
        e1 = {
            "strategy_id": "native_scale1",
            "strategy_scale": 1,
            "strategy_shift_k": 0,
            "dataset": "sensor",
            "n_rows": 1000,
            "reason": "inapplicable_overflow",
            "details": "overflow",
        }
        e2 = {
            "strategy_id": "per_dataset",
            "strategy_scale": 208,
            "strategy_shift_k": 0,
            "dataset": "heavy_tailed",
            "n_rows": 1000,
            "reason": "inapplicable_overflow",
            "details": "overflow",
        }
        log_skipped(csv_path, e1)
        log_skipped(csv_path, e2)

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 entries
        assert "heavy_tailed" in lines[2]
        assert "per_dataset" in lines[2]


# ── Main entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    test_uint64_to_planes_roundtrip()
    test_uint64_to_planes_shifted_values()
    test_load_params()
    test_load_params_missing_strategy()
    test_load_params_missing_dataset()
    test_get_strategy_params()
    test_make_artifact_dir_native()
    test_make_artifact_dir_per_dataset()
    test_make_artifact_dir_hybrid()
    test_convert_hybrid_tiny()
    test_convert_hybrid_full_roundtrip()
    test_convert_hybrid_overflow_skip()
    test_log_skipped_creates_file()
    test_log_skipped_append()
    print("All tests pass.")
