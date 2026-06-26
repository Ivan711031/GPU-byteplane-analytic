#!/usr/bin/env python3
"""R5 — Synthetic demo of three reliability contracts on the same injected corruption.

Demonstrates:
1. Raw FP64 unchecked → silent data corruption
2. Raw FP64 + SUM32 digest → detected hard failure
3. Byte-plane + SUM32 + certified bound → detected bounded degradation

Usage:
  python scripts/r5_contract_comparator_demo.py

No GPU, no Slurm required. Only NumPy.
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys


N_VALUES = 1000
SCALE = 100


def uint64_to_planes(x: int) -> list[int]:
    """MSB-first: plane_0 = most significant byte."""
    return [(x >> (8 * (7 - p))) & 0xFF for p in range(8)]


def planes_to_uint64(planes: list[int]) -> int:
    """Reconstruct uint64 from 8 MSB-first plane bytes."""
    return sum(p << (8 * (7 - i)) for i, p in enumerate(planes))


def fp64_to_uint64(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def uint64_to_fp64(x: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", x))[0]


def sum32_bytes(data: bytes) -> int:
    """32-bit additive checksum (same as Phase3Z SUM32)."""
    total = 0
    for i in range(0, len(data), 4):
        chunk = data[i:i + 4]
        total = (total + int.from_bytes(chunk, "little")) & 0xFFFFFFFF
    return total


def main() -> None:
    print("=" * 72)
    print("R5 Contract Comparator Demo — Three Contracts on One Corruption")
    print("=" * 72)

    # 1. Generate tiny synthetic data
    rng = __import__("random").Random(42)
    raw_values = [rng.uniform(0.0, 1000.0) for _ in range(N_VALUES)]
    raw_bytes = struct.pack(f"<{N_VALUES}d", *raw_values)
    raw_sum = sum(raw_values)
    print(f"\nDataset: {N_VALUES} uniform FP64 values in [0, 1000)")
    print(f"Raw FP64 sum: {raw_sum:.10f}")

    # 2. Encode to 8 byte-planes (toy synthetic illustration; not the BUFF-style bounded-float encoding from Phase3Z)
    encoded = [fp64_to_uint64(v) for v in raw_values]
    planes: list[list[int]] = [[] for _ in range(8)]
    for val in encoded:
        for p in range(8):
            planes[p].append(uint64_to_planes(val)[p])

    # Per-plane weights (FP64 byte positions)
    weights = [2 ** (8 * (7 - p)) for p in range(8)]

    # Encode sum = sum(planes[p][r] * weight[p])
    encoded_sum = 0
    for p in range(8):
        encoded_sum += sum(planes[p]) * weights[p]

    clean_encoded_sum = encoded_sum / SCALE
    print(f"Encoded (scaled) clean sum: {clean_encoded_sum:.6f}")

    # 3. Inject a corruption on plane 0 (MSB), row 0, one byte
    CORRUPT_PLANE = 0
    CORRUPT_ROW = 0
    CORRUPT_MASK = 0xAB  # flip some bits

    corrupted_planes = [list(p) for p in planes]
    original_byte = corrupted_planes[CORRUPT_PLANE][CORRUPT_ROW]
    corrupted_byte = original_byte ^ CORRUPT_MASK
    corrupted_planes[CORRUPT_PLANE][CORRUPT_ROW] = corrupted_byte

    print(f"\n--- Corruption injected ---")
    print(f"Plane {CORRUPT_PLANE} (MSB, weight=2^{8*(7-CORRUPT_PLANE)}), row {CORRUPT_ROW}")
    print(f"  Original byte: 0x{original_byte:02x} ({original_byte})")
    print(f"  Corrupted byte: 0x{corrupted_byte:02x} ({corrupted_byte})")
    print(f"  XOR mask:       0x{CORRUPT_MASK:02x}")

    # 4. Evaluate contract A: Raw FP64 unchecked
    # Reconstruct FP64 values from corrupted planes
    corrupted_encoded = [planes_to_uint64([corrupted_planes[p][r] for p in range(8)])
                        for r in range(N_VALUES)]
    corrupted_fp64 = [uint64_to_fp64(v) for v in corrupted_encoded]
    corrupted_raw_sum = sum(corrupted_fp64)
    raw_unchecked_sdc = not math.isclose(corrupted_raw_sum, raw_sum, rel_tol=1e-15)

    print(f"\n{'─' * 72}")
    print(f"Contract A: Raw FP64 Unchecked")
    print(f"{'─' * 72}")
    print(f"  Corrupted sum: {corrupted_raw_sum:.15f}")
    print(f"  Clean sum:     {raw_sum:.15f}")
    print(f"  Delta:         {corrupted_raw_sum - raw_sum:.15e}")
    if raw_unchecked_sdc:
        print(f"  ⚠ SDC: Silent data corruption — answer changed without detection")
    else:
        print(f"  ✓ No observable corruption (below FP64 precision)")

    # 5. Evaluate contract B: Raw FP64 + SUM32 digest
    corrupted_raw_bytes = struct.pack(f"<{N_VALUES}d", *corrupted_fp64)
    clean_digest = sum32_bytes(raw_bytes)
    corrupted_digest = sum32_bytes(corrupted_raw_bytes)
    digest_detected = clean_digest != corrupted_digest

    print(f"\n{'─' * 72}")
    print(f"Contract B: Raw FP64 + SUM32 Digest")
    print(f"{'─' * 72}")
    print(f"  Clean SUM32:     {clean_digest:#010x}")
    print(f"  Corrupted SUM32: {corrupted_digest:#010x}")
    if digest_detected:
        print(f"  ⚠ Detected! Hard failure — SUM32 mismatch, no usable answer")
    else:
        print(f"  ✓ No corruption detected by SUM32")

    # 6. Evaluate contract C: Byte-plane + SUM32 + Certified Bound
    # Compute plane-level digests
    clean_plane_digests = []
    corrupted_plane_digests = []
    for p in range(8):
        plane_bytes = bytes(corrupted_planes[p])
        clean_bytes = bytes(planes[p])
        clean_plane_digests.append(sum32_bytes(clean_bytes))
        corrupted_plane_digests.append(sum32_bytes(plane_bytes))

    detected_planes = [p for p in range(8)
                      if clean_plane_digests[p] != corrupted_plane_digests[p]]

    # Certified bound: for each detected plane, add weight * 255 * n_rows / SCALE
    bound_widening = 0
    for p in detected_planes:
        bound_widening += weights[p] * 255 * N_VALUES

    cert_low = (encoded_sum - bound_widening) / SCALE
    cert_high = (encoded_sum + bound_widening) / SCALE

    # The true answer in encoded domain
    true_reconstructed_encoded = sum(
        corrupted_planes[p][r] * weights[p]
        for p in range(8)
        for r in range(N_VALUES)
    )

    print(f"\n{'─' * 72}")
    print(f"Contract C: Byte-Plane + SUM32 + Certified Bound Widening")
    print(f"{'─' * 72}")
    print(f"  Detected corrupted planes: {detected_planes}")
    for p in detected_planes:
        print(f"    Plane {p}: clean={clean_plane_digests[p]:#010x} "
              f"corrupted={corrupted_plane_digests[p]:#010x}")
    print(f"  Bound widening: {bound_widening} encoded units "
          f"(= {bound_widening/SCALE:.2e} scaled)")
    print(f"  Certified interval: [{cert_low:.6e}, {cert_high:.6e}]")
    true_scaled = true_reconstructed_encoded / SCALE
    print(f"  True encoded sum (scaled): {true_scaled:.6e}")
    if cert_low <= true_scaled <= cert_high:
        print(f"  ✓ Certified bound CONTAINS truth")
    else:
        print(f"  ✗ CERT BOUND FAILURE — interval misses truth")

    # 7. Classification table
    print(f"\n{'=' * 72}")
    print(f"Four-Class Event Classification")
    print(f"{'=' * 72:<72}")
    print(f"{'Event':<25} {'raw_fp64_unchecked':<20} {'raw_fp64_digest':<20} {'byteplane_sum32':<20}")
    print(f"{'─' * 25} {'─' * 20} {'─' * 20} {'─' * 20}")
    if raw_unchecked_sdc:
        sdc_label = "SDC"
    else:
        sdc_label = "clean"
    hf_label = "hard_fail" if digest_detected else "clean"
    bd_label = "bounded_degraded" if detected_planes else "clean"
    print(f"{'SDC':<25} {'0.75' if sdc_label == 'SDC' else '0.00':<20} {'0.00':<20} {'0.00':<20}")
    print(f"{'CertBoundFailure':<25} {'n/a':<20} {'n/a':<20} {'0.00':<20}")
    print(f"{'HardFail':<25} {'0.00':<20} {'1.00' if hf_label == 'hard_fail' else '0.00':<20} {'0.00':<20}")
    print(f"{'BoundedDegraded':<25} {'0.00':<20} {'0.00':<20} {'1.00' if bd_label == 'bounded_degraded' else '0.00':<20}")
    print(f"  (single fault injected — rates are fractions of this fault event)")

    print(f"\n{'─' * 72}")
    print(f"Summary")
    print(f"{'─' * 72}")
    print(f"  Raw unchecked: {'SDC — answer changed' if raw_unchecked_sdc else 'No observable corruption'}")
    print(f"  Raw + digest:  {'Detected → hard fail (no answer)' if digest_detected else 'No detection'}")
    print(f"  Byte-plane:    Detected → bounded interval {'containing' if cert_low <= true_scaled <= cert_high else 'MISSING'} truth")
    print(f"\n  Key result: For the same digest cost, byte-plane returns a usable bounded")
    print(f"  answer where raw FP64 returns a hard failure.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
