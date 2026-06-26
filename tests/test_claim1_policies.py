"""Deterministic unit tests for Claim 1 policy allocator.

P0 gating: tests must pass before any formal pilot run.
"""

from scripts.nmr_claim1_realistic_campaign import (
    FaultEvent,
    make_graded_policy,
    make_uniform_policy,
    build_policy_catalogue,
    _SEVERITY_SWEEP,
    _FAULT_FAMILIES,
)


def test_graded_B0():
    """B=0: no extra copies, all planes at base r=1."""
    e = make_graded_policy(0)
    assert e == [0, 0, 0, 0, 0, 0, 0, 0]
    assert sum(e) == 0


def test_graded_B1():
    """B=1: first extra goes to plane 0 (r=2)."""
    e = make_graded_policy(1)
    assert e == [1, 0, 0, 0, 0, 0, 0, 0]
    assert sum(e) == 1


def test_graded_B2():
    """B=2: plane 0 gets second extra (r=3)."""
    e = make_graded_policy(2)
    assert e == [2, 0, 0, 0, 0, 0, 0, 0]
    assert sum(e) == 2


def test_graded_B3():
    """B=3: plane 0 at r=3, plane 1 gets first extra (r=2)."""
    e = make_graded_policy(3)
    assert e == [2, 1, 0, 0, 0, 0, 0, 0]
    assert sum(e) == 3


def test_graded_B4():
    """B=4: planes 0-1 at r=3."""
    e = make_graded_policy(4)
    assert e == [2, 2, 0, 0, 0, 0, 0, 0]
    assert sum(e) == 4


def test_graded_B16():
    """B=16: all planes at r=3 (max)."""
    e = make_graded_policy(16)
    assert e == [2, 2, 2, 2, 2, 2, 2, 2]
    assert sum(e) == 16


def test_graded_invariants():
    """Every B in 0..16 satisfies sum=r_max<=3."""
    for B in range(0, 17):
        e = make_graded_policy(B)
        r = [1 + v for v in e]
        assert sum(e) == B, f"B={B}: sum(extras)={sum(e)} != {B}"
        assert max(r) <= 3, f"B={B}: max(r)={max(r)} > 3"
        assert all(v >= 0 for v in e)
        # monotonic: increasing B never decreases any plane's extras
        if B > 0:
            prev = make_graded_policy(B - 1)
            assert all(e[p] >= prev[p] for p in range(8)), (
                f"B={B}: non-monotonic vs B={B-1}"
            )


def test_graded_max_cap():
    """No plane exceeds r=3 even at B>16."""
    e = make_graded_policy(100)
    r = [1 + v for v in e]
    assert all(v <= 2 for v in e), f"extras exceed cap: {e}"
    assert max(r) <= 3


def test_uniform_basics():
    """Uniform policy distribution is even."""
    for B in [0, 1, 8, 16]:
        e = make_uniform_policy(B)
        assert sum(e) == B, f"B={B}: sum={sum(e)}"
        assert min(e) >= 0
        assert max(e) - min(e) <= 1, f"B={B}: spread > 1"


def test_catalogue_20():
    """Policy catalogue has exactly 20 entries with correct names."""
    cats = build_policy_catalogue()
    assert len(cats) == 20

    names = [name for name, _, _ in cats]
    assert "uniform_full_r1" in names
    assert "uniform_full_r2" in names
    assert "uniform_full_r3" in names
    for B in range(0, 17):
        assert f"graded_B{B}" in names, f"Missing graded_B{B}"

    assert len(names) == len(set(names)), "Duplicate policy names"


def test_catalogue_budget_match():
    """At matched B, uniform and graded differ but extras sum matches."""
    cats = build_policy_catalogue()
    graded_by_B = {}
    for name, kind, extras in cats:
        if kind == "graded":
            graded_by_B[sum(extras)] = (name, extras)

    u2_extras = [e for n, k, e in cats if n == "uniform_full_r2"][0]
    g8 = graded_by_B[8]
    assert sum(u2_extras) == sum(g8[1])
    assert u2_extras != g8[1], (
        "uniform_full_r2 and graded_B8 should differ at same B"
    )


def test_catalogue_deterministic():
    """Multiple calls produce identical catalogues."""
    cats1 = build_policy_catalogue()
    cats2 = build_policy_catalogue()
    assert cats1 == cats2


def test_severity_sweep_all_families():
    """Every family in _FAULT_FAMILIES has at least 1 sweep entry."""
    for family in _FAULT_FAMILIES:
        assert family in _SEVERITY_SWEEP, f"Missing severity sweep for {family}"
        assert len(_SEVERITY_SWEEP[family]) >= 1


def test_severity_sweep_deterministic():
    """Severity sweep entries are stable across calls (same list structure)."""
    from scripts.nmr_claim1_realistic_campaign import _SEVERITY_SWEEP as s1
    from importlib import reload
    import scripts.nmr_claim1_realistic_campaign as mod
    reload(mod)
    s2 = mod._SEVERITY_SWEEP
    assert s1 == s2


def test_total_sweep_cells():
    """Total severity cells per dataset/seed/policy = 11."""
    total = sum(len(knobs) for knobs in _SEVERITY_SWEEP.values())
    assert total == 11, f"Expected 11 sweep cells, got {total}"


# ---------------------------------------------------------------------------
# r2/r3 detector metric tests
# ---------------------------------------------------------------------------

def test_parse_r_vector():
    from scripts.nmr_claim1_realistic_campaign import _parse_r_vector
    assert _parse_r_vector("[1, 2, 1, 1, 1, 1, 1, 1]") == [1, 2, 1, 1, 1, 1, 1, 1]
    assert _parse_r_vector("[3, 3, 3, 3, 3, 3, 3, 3]") == [3, 3, 3, 3, 3, 3, 3, 3]
    assert _parse_r_vector("[]") == []


def test_detector_metrics_r2_on_faulted_plane():
    """r2_disagreement_rate counts rows where faulted plane has r>=2."""
    from scripts.nmr_claim1_realistic_campaign import compute_detector_metrics
    rows = [
        {"r_vector": "[1, 1, 1, 1, 1, 1, 1, 1]", "faulted_planes": "[0]",
         "detected": True, "outcome": "certified_degraded", "event_count": 1},
        {"r_vector": "[2, 1, 1, 1, 1, 1, 1, 1]", "faulted_planes": "[0]",
         "detected": True, "outcome": "certified_degraded", "event_count": 1},
    ]
    m = compute_detector_metrics(rows)
    # Row 0: faulted plane 0 has r=1 (<2) → excluded from r2_rows
    # Row 1: faulted plane 0 has r=2 (>=2) → included, detected=True
    # r2_n = 1, r2_disagreement = 1.0
    assert m["r2_disagreement_rate"] == 1.0, f"got {m['r2_disagreement_rate']}"
    # r3_applicable_cells = 0 → r3_majority_repair_rate = None
    assert m["r3_majority_repair_rate"] is None, f"got {m['r3_majority_repair_rate']}"


def test_detector_metrics_r3_on_faulted_plane():
    """r3_majority_repair_rate counts rows where faulted plane has r>=3."""
    from scripts.nmr_claim1_realistic_campaign import compute_detector_metrics
    rows = [
        {"r_vector": "[2, 1, 1, 1, 1, 1, 1, 1]", "faulted_planes": "[0]",
         "detected": True, "outcome": "certified_degraded", "event_count": 1},
        {"r_vector": "[3, 1, 1, 1, 1, 1, 1, 1]", "faulted_planes": "[0]",
         "detected": True, "outcome": "exact_correct", "event_count": 1},
    ]
    m = compute_detector_metrics(rows)
    # Row 0: faulted plane 0 has r=2 (<3) → excluded from r3_rows
    # Row 1: faulted plane 0 has r=3 (>=3) → included, outcome=exact_correct
    # r3_n = 1, r3_repair = 1.0
    assert m["r3_majority_repair_rate"] == 1.0, f"got {m['r3_majority_repair_rate']}"
    assert m["r2_disagreement_rate"] == 1.0


def test_detector_metrics_different_plane_not_counted():
    """Fault on plane 3 should NOT trigger r2/r3 count if plane 3 has r=1,
    even if plane 0 has r=3."""
    from scripts.nmr_claim1_realistic_campaign import compute_detector_metrics
    rows = [
        {"r_vector": "[3, 3, 3, 1, 1, 1, 1, 1]", "faulted_planes": "[3]",
         "detected": True, "outcome": "certified_degraded", "event_count": 1},
    ]
    m = compute_detector_metrics(rows)
    # faulted plane 3 has r=1 (<2) → excluded from both r2_rows and r3_rows
    # r2_applicable_cells = 0 → r2_disagreement_rate = None
    # r3_applicable_cells = 0 → r3_majority_repair_rate = None
    assert m["r2_disagreement_rate"] is None, f"got {m['r2_disagreement_rate']}"
    assert m["r3_majority_repair_rate"] is None, f"got {m['r3_majority_repair_rate']}"


# ---------------------------------------------------------------------------
# Replication-only outcome classifier tests
# ---------------------------------------------------------------------------


def _make_fault_event(plane: int = 0) -> FaultEvent:
    return FaultEvent(plane=plane, offset=0, mask=0xFF)


def _make_ppo(r: int, diff_bytes: int = 0) -> dict:
    return {"r": r, "diff_bytes": diff_bytes, "detected": diff_bytes > 0,
            "disagreement": diff_bytes > 0 if r == 2 else False}


def _make_ppc(plane: int, r: int, per_plane_outcome: str,
              diff_bytes: int = 0) -> dict:
    return {"plane": plane, "r": r, "diff_bytes": diff_bytes,
            "per_plane_outcome": per_plane_outcome}


def test_rep_outcome_r1_fault_silent_wrong():
    """r=1 on faulted plane, delivered != clean → silent_wrong."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [1, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(1, diff_bytes=1)]
    ppc = [_make_ppc(0, 1, "silent_wrong", diff_bytes=1)]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "silent_wrong", f"got {oc}"


def test_rep_outcome_r2_disagreement_detected_unavailable():
    """r=2 on faulted plane, replicas differ → detected_unavailable."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [2, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(2, diff_bytes=1)]
    ppc = [_make_ppc(0, 2, "detected_unavailable", diff_bytes=1)]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "detected_unavailable", f"got {oc}"


def test_rep_outcome_r2_twin_corruption_silent_wrong():
    """r=2, replicas identical but wrong (twin corruption) → silent_wrong."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [2, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(2, diff_bytes=0)]
    ppc = [_make_ppc(0, 2, "silent_wrong", diff_bytes=0)]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "silent_wrong", f"got {oc}"


def test_rep_outcome_r3_majority_recovers():
    """r=3, majority vote recovers correct → majority_recovered."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [3, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(3, diff_bytes=0)]
    ppc = [_make_ppc(0, 3, "majority_recovered")]
    oc = classify_replication_outcome(5.0, 5.0, rv, events, ppo, ppc)
    assert oc == "majority_recovered", f"got {oc}"


def test_rep_outcome_no_fault_exact_correct():
    """No fault events → exact_correct."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    oc = classify_replication_outcome(5.0, 5.0, [1]*8, [], [])
    assert oc == "exact_correct", f"got {oc}"


# ---------------------------------------------------------------------------
# Per-plane + query combine tests (Issue #285 combine logic)
# ---------------------------------------------------------------------------

def test_combine_mixed_r1_sw_and_r2_du_yields_sw():
    """r=1 silent_wrong + r=2 detected_unavailable → query silent_wrong.
    SW priority over DU."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [2, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0), _make_fault_event(1)]
    ppo = [_make_ppo(2, diff_bytes=1), _make_ppo(1, diff_bytes=1)]
    ppc = [
        _make_ppc(0, 2, "detected_unavailable", diff_bytes=1),
        _make_ppc(1, 1, "silent_wrong", diff_bytes=1),
    ]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "silent_wrong", f"got {oc} (should be silent_wrong: SW>DU)"


def test_combine_mixed_r1_sw_and_r3_wrong_majority():
    """r=1 silent_wrong + r=3 wrong majority vote → query silent_wrong.
    SW priority over everything."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [3, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0), _make_fault_event(1)]
    ppo = [_make_ppo(3, diff_bytes=1), _make_ppo(1, diff_bytes=1)]
    ppc = [
        _make_ppc(0, 3, "silent_wrong"),  # F7-style: 2-of-3 wrong majority
        _make_ppc(1, 1, "silent_wrong", diff_bytes=1),
    ]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "silent_wrong", f"got {oc}"


def test_combine_pure_r2_disagreement_detected_unavailable():
    """Only r=2 disagreement, all other planes correct → query detected_unavailable."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [2, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(2, diff_bytes=1)]
    ppc = [_make_ppc(0, 2, "detected_unavailable", diff_bytes=1)]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "detected_unavailable", f"got {oc}"


def test_combine_pure_r3_correct_majority():
    """Only r=3 correct majority → query majority_recovered."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [3, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(3, diff_bytes=0)]
    ppc = [_make_ppc(0, 3, "majority_recovered")]
    oc = classify_replication_outcome(5.0, 5.0, rv, events, ppo, ppc)
    assert oc == "majority_recovered", f"got {oc}"


def test_combine_r2_twin_corruption_silent_wrong():
    """r=2 twin corruption → query silent_wrong (not detected_unavailable)."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [2, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(2, diff_bytes=0)]
    ppc = [_make_ppc(0, 2, "silent_wrong", diff_bytes=0)]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "silent_wrong", f"got {oc}"


def test_combine_f7_on_graded_B2_silent_wrong():
    """F7 on graded_B2 (r=3 on plane 0): 2-of-3 wrong majority vote.
    Plane outcome must be silent_wrong, query must be silent_wrong."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [3, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(3, diff_bytes=1)]
    ppc = [_make_ppc(0, 3, "silent_wrong")]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "silent_wrong", f"got {oc} (F7 on r=3: wrong majority → SW)"


def test_combine_r1_correct_no_fault_exact_correct():
    """r=1 faulted plane but delivered == clean → exact_correct."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [1, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(1, diff_bytes=0)]
    ppc = [_make_ppc(0, 1, "exact_correct", diff_bytes=0)]
    oc = classify_replication_outcome(5.0, 5.0, rv, events, ppo, ppc)
    assert oc == "exact_correct", f"got {oc}"


def test_combine_r2_agreement_correct_exact_correct():
    """r=2, replicas agree AND correct → exact_correct."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [2, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(2, diff_bytes=0)]
    ppc = [_make_ppc(0, 2, "exact_correct", diff_bytes=0)]
    oc = classify_replication_outcome(5.0, 5.0, rv, events, ppo, ppc)
    assert oc == "exact_correct", f"got {oc}"


def test_combine_r2_du_r3_mr_yields_mr():
    """r=2 DU + r=3 MR → query MR (MR < DU < SW priority, so DU wins).
    Actually: SW > DU > MR: no SW. DU exists. → detected_unavailable."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [3, 2, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0), _make_fault_event(1)]
    ppo = [_make_ppo(3, diff_bytes=0), _make_ppo(2, diff_bytes=1)]
    ppc = [
        _make_ppc(0, 3, "majority_recovered"),
        _make_ppc(1, 2, "detected_unavailable", diff_bytes=1),
    ]
    oc = classify_replication_outcome(5.0, 5.0, rv, events, ppo, ppc)
    assert oc == "detected_unavailable", f"got {oc} (DU > MR)"


def test_combine_r3_wrong_majority_no_other_faults():
    """r=3 wrong majority (F7-style), no other faults → query silent_wrong."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [3, 1, 1, 1, 1, 1, 1, 1]
    events = [_make_fault_event(0)]
    ppo = [_make_ppo(3, diff_bytes=1)]
    ppc = [_make_ppc(0, 3, "silent_wrong")]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "silent_wrong", f"got {oc}"


def test_combine_multiple_du_no_sw():
    """Multiple r=2 disagreements but no silent_wrong → query detected_unavailable."""
    from scripts.nmr_claim1_realistic_campaign import classify_replication_outcome
    rv = [2, 2, 2, 2, 2, 2, 2, 2]
    events = [_make_fault_event(0), _make_fault_event(1)]
    ppo = [_make_ppo(2, diff_bytes=1), _make_ppo(2, diff_bytes=1)]
    ppc = [
        _make_ppc(0, 2, "detected_unavailable", diff_bytes=1),
        _make_ppc(1, 2, "detected_unavailable", diff_bytes=1),
    ]
    oc = classify_replication_outcome(5.0, 3.0, rv, events, ppo, ppc)
    assert oc == "detected_unavailable", f"got {oc}"


# ---------------------------------------------------------------------------
# Replication-only outcome + metrics computation tests
# ---------------------------------------------------------------------------


def test_rep_detected_unavailable_excluded_from_point_metrics():
    """Rows with outcome=detected_unavailable are excluded from point-answer
    error metrics (err_fault_conditioned_user_observed, relative_error_p*,
    decision_flip_rate, catastrophic_error_rate)."""
    from scripts.nmr_claim1_realistic_campaign import compute_replication_primary_metrics
    rows = [
        {"outcome": "exact_correct", "relative_error": 0.0, "decision_flip": False},
        {"outcome": "detected_unavailable", "relative_error": 0.5, "decision_flip": True},
        {"outcome": "silent_wrong", "relative_error": 2.0, "decision_flip": True},
    ]
    m = compute_replication_primary_metrics(rows)
    # Only 2 rows enter point-answer stats: exact_correct (rel_err=0) and silent_wrong (rel_err=2.0)
    # detected_unavailable is excluded
    # p50 of [0.0, 2.0] at index n//2=1 → 2.0
    assert m["err_fault_conditioned_user_observed"] == 1.0, f"got {m['err_fault_conditioned_user_observed']}"
    assert m["relative_error_p50"] == 2.0, f"got {m['relative_error_p50']}"
    assert m["decision_flip_rate_at_tau"] == 0.5, f"got {m['decision_flip_rate_at_tau']}"
    # detected_unavailable IS counted in outcome rates
    assert m["exact_correct_rate"] == 1.0 / 3, f"got {m['exact_correct_rate']}"
    assert m["detected_unavailable_rate"] == 1.0 / 3, f"got {m['detected_unavailable_rate']}"
    assert m["silent_wrong_rate"] == 1.0 / 3, f"got {m['silent_wrong_rate']}"


def test_rep_composite_metrics():
    """correct_answer_rate = (exact_correct + majority_recovered) / n;
    safe_answer_rate = 1 - silent_wrong / n."""
    from scripts.nmr_claim1_realistic_campaign import compute_replication_composite_metrics
    rows = [
        {"outcome": "exact_correct"},
        {"outcome": "majority_recovered"},
        {"outcome": "detected_unavailable"},
        {"outcome": "silent_wrong"},
    ]
    m = compute_replication_composite_metrics(rows)
    assert m["correct_answer_rate"] == 0.5, f"got {m['correct_answer_rate']}"
    assert m["safe_answer_rate"] == 0.75, f"got {m['safe_answer_rate']}"

