import numpy as np

from app.services.inference import (
    late_fusion_weighted_average,
    early_fusion_concat_normalize,
    compare_fusion_vs_individuals,
)


def test_late_fusion_weighted_average_basic():
    v = np.array([0.1, 0.9], dtype="float32")
    t = np.array([0.6, 0.4], dtype="float32")
    pred, conf, comb = late_fusion_weighted_average(v, t, w_visual=0.7, w_tab=0.3)
    # combined = 0.7*v + 0.3*t
    expected = 0.7 * v + 0.3 * t
    expected = expected / expected.sum()
    assert np.allclose(comb, expected, atol=1e-6)
    assert pred in (0, 1)
    assert abs(conf - comb[pred]) < 1e-6


def test_early_fusion_concat_normalize_shape():
    v = np.array([1.0, 2.0, 3.0])
    t = np.array([4.0, 5.0])
    fused = early_fusion_concat_normalize(v, t)
    assert fused.shape == (1, 5)
    # Each sub-block should be L2-normalized
    v_part = fused[0, :3]
    t_part = fused[0, 3:]
    assert abs(np.linalg.norm(v_part) - 1.0) < 1e-6
    assert abs(np.linalg.norm(t_part) - 1.0) < 1e-6


def test_compare_fusion_vs_individuals_outputs():
    fusion = np.array([0.6, 0.4])
    visual = np.array([0.5, 0.5])
    tab = np.array([0.4, 0.6])
    res = compare_fusion_vs_individuals(fusion, visual, tab)
    assert set(res.keys()) >= {"verdict", "fusion_top", "visual_top", "tab_top"}
    assert isinstance(res["fusion_top"], float)