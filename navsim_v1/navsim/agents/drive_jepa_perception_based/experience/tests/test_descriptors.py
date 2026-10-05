"""Unit tests for experience descriptors — synthetic shapely polygons only."""

import numpy as np
import pytest
from shapely.geometry import box

from navsim.agents.drive_jepa_perception_based.experience.descriptors import (
    DESCRIPTOR_FIELDS,
    ITYPE_CROSSING,
    ITYPE_NONE,
    ITYPE_ONCOMING,
    ITYPE_SAME_DIR,
    KNN_NUMERIC_FIELDS,
    NUM_DESCRIPTOR_FIELDS,
    TIMING_FIELDS,
    compute_interaction_descriptor,
    descriptor_feature_names,
    descriptor_feature_vector,
    descriptor_sort_key,
    select_top_m_vehicles,
    wrap_angle,
)

DT = 0.1
T = 10  # 11 timesteps


def ego_seq(x0=0.0, y0=0.0, vx=5.0):
    """Ego box (2x2) moving +x at vx m/s for T+1 steps."""
    return [box(x0 + vx * DT * t - 1, y0 - 1, x0 + vx * DT * t + 1, y0 + 1) for t in range(T + 1)]


def j_seq(x=6.0, y=0.0, dxdt=0.0, dydt=0.0):
    """Vehicle j box moving linearly; always present."""
    return [
        box(x + dxdt * DT * t - 1, y + dydt * DT * t - 1, x + dxdt * DT * t + 1, y + dydt * DT * t + 1)
        for t in range(T + 1)
    ]


def j_seq_absent_until(t0):
    """Vehicle j present only from step t0 (fixed position at x=6)."""
    out = [None] * (T + 1)
    for t in range(t0, T + 1):
        out[t] = box(5, -1, 7, 1)
    return out


class TestConflictBasics:
    def test_no_j_present(self):
        d = compute_interaction_descriptor(ego_seq(), [None] * (T + 1), 0.0, 0.0)
        assert d["conflict"] == 0.0
        assert d["itype"] == ITYPE_NONE
        assert np.isnan(d["min_dist"])
        assert np.isnan(d["dt_enter"])
        assert d["k_in_censored"] == 0.0

    def test_j_far_away_no_conflict(self):
        d = compute_interaction_descriptor(ego_seq(), j_seq(x=100.0), 0.0, 0.0)
        assert d["conflict"] == 0.0
        assert d["itype"] == ITYPE_NONE
        # min_dist still computed (they are co-present) even without conflict
        assert np.isfinite(d["min_dist"]) and d["min_dist"] > 50

    def test_head_on_overlap_conflict(self):
        # ego at x=0.., j static at x=6 overlapping ego path: ego enters swept_j
        # at t where ego x-range hits [5,7] -> ego right edge (vx*t+1) >= 5 -> t >= 8
        d = compute_interaction_descriptor(ego_seq(vx=5.0), j_seq(x=6.0), 0.0, 0.0)
        assert d["conflict"] == 1.0
        assert d["att_collision"] == 0.0
        assert d["min_dist"] == 0.0
        assert np.isfinite(d["t_k_in"]) and np.isfinite(d["t_j_in"])
        # ego reaches j's static swept box late; j inside ego's swept volume from t=0
        assert d["t_j_in"] == 0.0
        assert d["j_in_censored"] == 1.0
        assert d["dt_enter"] > 0  # ego enters conflict after j
        assert d["overlap"] == 1.0
        assert d["pet"] == 0.0

    def test_attribution_flags(self):
        d = compute_interaction_descriptor(
            ego_seq(), j_seq(x=6.0), 0.0, 0.0, att_collision=True, att_ttc=True
        )
        assert d["att_collision"] == 1.0 and d["att_ttc"] == 1.0

    def test_no_overlap_pet(self):
        # j crosses ego's swept path briefly at the very start (j present only t=0..2,
        # positioned where ego will be later -> j_occ early, ego_occ late)
        jp = [box(3.0, -1, 5.0, 1) if t <= 2 else None for t in range(T + 1)]
        d = compute_interaction_descriptor(ego_seq(vx=5.0), jp, 0.0, 0.0)
        assert d["conflict"] == 1.0
        assert d["overlap"] == 0.0
        # j was in ego's swept path from t=0 (j_in censored), ego enters swept_j at t~4
        assert d["j_in_censored"] == 1.0
        assert d["k_in_censored"] == 0.0
        assert np.isfinite(d["pet"])
        # pet = max(t_k_in, t_j_in) - min(t_k_out, t_j_out); t_j_out = 0.2
        assert d["pet"] == pytest.approx(d["t_k_in"] - 0.2)

    def test_multi_entry(self):
        # j toggles present/absent so j_occ has 2 runs; swept_j covers x in [5,7]
        # and j polygon intersects ego sweep at its present steps only.
        jp = []
        for t in range(T + 1):
            if t in (0, 1, 5, 6, 7):
                jp.append(box(5, -1, 7, 1))
            else:
                jp.append(box(100, -1, 102, 1))
        d = compute_interaction_descriptor(ego_seq(vx=5.0), jp, 0.0, 0.0)
        assert d["conflict"] == 1.0
        assert d["multi_entry"] == 1.0

    def test_k_out_censored(self):
        # ego still inside swept_j at final step (ego only reaches j at t=10)
        d = compute_interaction_descriptor(ego_seq(vx=5.0), j_seq(x=5.5), 0.0, 0.0)
        assert d["conflict"] == 1.0
        assert d["k_out_censored"] == 1.0
        # and a faster ego leaves swept_j before the end -> not censored
        d2 = compute_interaction_descriptor(ego_seq(vx=2.0), j_seq(x=4.5), 0.0, 0.0)
        assert d2["k_out_censored"] == 0.0


class TestItype:
    def _conflict_desc(self, j_heading):
        return compute_interaction_descriptor(ego_seq(vx=5.0), j_seq(x=6.0), 0.0, j_heading)

    def test_same_dir(self):
        assert self._conflict_desc(0.1)["itype"] == ITYPE_SAME_DIR

    def test_oncoming(self):
        assert self._conflict_desc(np.pi)["itype"] == ITYPE_ONCOMING

    def test_crossing(self):
        assert self._conflict_desc(np.pi / 2)["itype"] == ITYPE_CROSSING

    def test_none_without_conflict(self):
        d = compute_interaction_descriptor(ego_seq(), j_seq(x=100.0), 0.0, np.pi)
        assert d["itype"] == ITYPE_NONE


class TestEgoFrameState:
    def test_rel_xy_heading_0(self):
        d = compute_interaction_descriptor(ego_seq(), j_seq(x=10.0, y=3.0), 0.0, 0.0, j_speed=4.0)
        # j center (10,3), ego center (0,0), heading 0 -> rel = (10,3)
        assert d["rel_x"] == pytest.approx(10.0)
        assert d["rel_y"] == pytest.approx(3.0)
        assert d["rel_heading"] == pytest.approx(0.0)
        assert d["speed"] == 4.0

    def test_rel_xy_heading_rotated(self):
        # ego heading +90deg: a j directly ahead in +x appears to the right (rel_y < 0)
        d = compute_interaction_descriptor(
            ego_seq(), j_seq(x=10.0, y=0.0), np.pi / 2, np.pi / 2, j_speed=1.0
        )
        assert d["rel_x"] == pytest.approx(0.0, abs=1e-6)
        assert d["rel_y"] == pytest.approx(-10.0, abs=1e-6)
        assert d["rel_heading"] == pytest.approx(0.0, abs=1e-6)

    def test_rel_absent_j0(self):
        d = compute_interaction_descriptor(ego_seq(), j_seq_absent_until(5), 0.0, 0.0)
        assert np.isnan(d["rel_x"]) and np.isnan(d["speed"])


class TestRanking:
    def test_att_collision_first(self):
        a = compute_interaction_descriptor(ego_seq(), j_seq(x=50.0), 0.0, 0.0, att_collision=True)
        b = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0, att_ttc=True)
        c = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0)
        order = select_top_m_vehicles([a, b, c], 3)
        assert order[0] == 0  # att_collision wins over att_ttc wins over plain conflict

    def test_conflict_beats_no_conflict(self):
        a = compute_interaction_descriptor(ego_seq(), j_seq(x=100.0), 0.0, 0.0)
        b = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0)
        assert select_top_m_vehicles([a, b], 1) == [1]

    def test_smaller_dt_enter_first(self):
        # both conflict: smaller |dt_enter| should rank first
        # early: j appears at t=5, ego enters swept_j at t=8 -> dt_enter = 0.3
        early = compute_interaction_descriptor(ego_seq(vx=5.0), j_seq_absent_until(5), 0.0, 0.0)
        # late: j static in ego path (j_in from t=0), ego enters at t=8 -> dt_enter = 0.8
        late = compute_interaction_descriptor(ego_seq(vx=5.0), j_seq(x=6.0), 0.0, 0.0)
        assert early["conflict"] == 1.0 and late["conflict"] == 1.0
        descs = [late, early]
        order = select_top_m_vehicles(descs, 2)
        assert order[0] == 1

    def test_m_truncates(self):
        descs = [
            compute_interaction_descriptor(ego_seq(), j_seq(x=6.0 + i), 0.0, 0.0)
            for i in range(6)
        ]
        assert len(select_top_m_vehicles(descs, 4)) == 4


class TestLeakageGuards:
    def test_att_flags_not_in_features(self):
        """att_collision/att_ttc must never enter the feature vector."""
        assert "att_collision" not in KNN_NUMERIC_FIELDS
        assert "att_ttc" not in KNN_NUMERIC_FIELDS
        assert "att_collision" not in TIMING_FIELDS
        assert "att_ttc" not in TIMING_FIELDS
        d_att = compute_interaction_descriptor(
            ego_seq(), j_seq(x=6.0), 0.0, 0.0, att_collision=True, att_ttc=True
        )
        d_plain = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0)
        for fields in (None, TIMING_FIELDS):
            v_att = descriptor_feature_vector(d_att, 5.0, fields)
            v_plain = descriptor_feature_vector(d_plain, 5.0, fields)
            assert np.allclose(
                np.nan_to_num(v_att), np.nan_to_num(v_plain)
            ), "feature vector must be identical with/without att flags"

    def test_noatt_ordering_ignores_attribution(self):
        """use_attribution=False ranks by conflict/timing only."""
        att_far = compute_interaction_descriptor(
            ego_seq(), j_seq(x=50.0), 0.0, 0.0, att_collision=True
        )
        conflict_near = compute_interaction_descriptor(
            ego_seq(), j_seq(x=6.0), 0.0, 0.0
        )
        # with attribution: the flagged (conflict-free!) vehicle still wins
        assert select_top_m_vehicles([att_far, conflict_near], 1) == [0]
        # without attribution: the conflicting vehicle wins
        assert select_top_m_vehicles([att_far, conflict_near], 1, use_attribution=False) == [1]

    def test_timing_excludes_near_label_fields(self):
        assert "min_dist" not in TIMING_FIELDS
        assert "overlap" not in TIMING_FIELDS
        d = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0, j_speed=1.0)
        v = descriptor_feature_vector(d, 5.0, TIMING_FIELDS)
        assert len(v) == len(descriptor_feature_names(TIMING_FIELDS))


class TestFeatureVector:
    def test_layout_and_size(self):
        d = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0, j_speed=2.0)
        v = descriptor_feature_vector(d, ego_speed=8.0)
        assert v.ndim == 1 and v.dtype == np.float32
        assert len(v) == len(descriptor_feature_names())
        assert v[-3:] == pytest.approx([1.0, 0.0, 0.0])  # SAME_DIR one-hot

    def test_nan_preserved(self):
        d = compute_interaction_descriptor(ego_seq(), [None] * (T + 1), 0.0, 0.0)
        v = descriptor_feature_vector(d)
        assert np.isnan(v).any()

    def test_schema_fields_complete(self):
        from navsim.agents.drive_jepa_perception_based.experience.descriptors import (
            DESCRIPTOR_FIELDS_EXT,
        )
        d = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0)
        assert set(d.keys()) == set(DESCRIPTOR_FIELDS_EXT)
        assert len(DESCRIPTOR_FIELDS) == NUM_DESCRIPTOR_FIELDS


class TestWrapAngle:
    def test_basic(self):
        assert wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
        assert wrap_angle(0.0) == 0.0
        assert wrap_angle(np.pi) == pytest.approx(-np.pi)


class TestVruExtension:
    def test_ext_fields_do_not_change_vehicle_fields(self):
        """DESCRIPTOR_FIELDS layout unchanged; EXT appends exactly 2 fields."""
        from navsim.agents.drive_jepa_perception_based.experience.descriptors import (
            ACLASS_BICYCLE,
            ACLASS_PEDESTRIAN,
            ACLASS_VEHICLE,
            DESCRIPTOR_FIELDS_EXT,
        )
        assert DESCRIPTOR_FIELDS_EXT[: len(DESCRIPTOR_FIELDS)] == DESCRIPTOR_FIELDS
        assert DESCRIPTOR_FIELDS_EXT[-2:] == ["agent_class", "emerging"]
        assert NUM_DESCRIPTOR_FIELDS == len(DESCRIPTOR_FIELDS)
        assert ACLASS_VEHICLE == 0 and ACLASS_PEDESTRIAN == 1 and ACLASS_BICYCLE == 2

    def test_emerging_flag_set(self):
        """Absent at t=0, present + conflicting later -> emerging=1."""
        d = compute_interaction_descriptor(
            ego_seq(), j_seq_absent_until(3), 0.0, 0.0,
            agent_class=1.0,
        )
        assert d["conflict"] == 1.0
        assert d["emerging"] == 1.0
        assert d["agent_class"] == 1.0
        # no t=0 state -> deployment fields stay NaN
        assert np.isnan(d["rel_x"])
        assert np.isnan(d["rel_heading"])
        assert np.isnan(d["speed"])

    def test_emerging_flag_requires_conflict(self):
        """t0-absent but never conflicting -> emerging=0."""
        polys = [None] * (T + 1)
        for t in range(3, T + 1):
            polys[t] = box(100, 99, 102, 101)
        d = compute_interaction_descriptor(ego_seq(), polys, 0.0, 0.0)
        assert d["conflict"] == 0.0
        assert d["emerging"] == 0.0

    def test_present_at_t0_not_emerging(self):
        d = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0)
        assert d["conflict"] == 1.0
        assert d["emerging"] == 0.0
        assert d["agent_class"] == 0.0

    def test_default_agent_class_vehicle(self):
        d = compute_interaction_descriptor(ego_seq(), j_seq(x=6.0), 0.0, 0.0)
        assert d["agent_class"] == 0.0
