"""Unit tests for CrossGuard ML preprocessing, NMS, zone analysis, and heuristics."""
import numpy as np
import pytest
from crossguard.pipeline import nms, postprocess, analyse, validate_zones, preprocess


def test_nms_class_independent_and_scaling():
    """Verify postprocessing NMS filtering, candidate selection, and resolution scaling."""
    boxes = np.array([[10, 10, 100, 100], [12, 12, 101, 101], [180, 180, 220, 250]], dtype=np.float32)
    scores = np.zeros((3, 6), np.float32)
    scores[0, 0] = .9
    scores[1, 0] = .8
    scores[1, 2] = .85
    scores[2, 5] = .95
    det = postprocess(boxes, scores, 640, 320)
    assert len(det) == 3
    assert {x['class'] for x in det} == {'person', 'car', 'truck'}
    person = next(x for x in det if x['class'] == 'person')
    assert person['box'] == [20, 10, 200, 100]


def test_empty_and_disabled_alerts():
    """Verify that zero-detection frames produce zero counts and disable zone alerts gracefully."""
    det = postprocess(np.zeros((5, 4)), np.zeros((5, 6)), 640, 480)
    counts, zone = analyse(det, None, 640, 480)
    assert sum(counts.values()) == 0 and zone['potential_conflict'] is None


def test_zone_cooccupancy_and_boundary():
    """Verify spatial zone co-occupancy logic and potential conflict detection between pedestrians and vehicles."""
    zones = validate_zones({
        'pedestrian_zone': [[0, 0], [.5, 0], [.5, 1], [0, 1]],
        'vehicle_zone': [[.5, 0], [1, 0], [1, 1], [.5, 1]]
    })
    person = {'class': 'person', 'box': [10, 10, 30, 80]}
    car = {'class': 'car', 'box': [60, 10, 90, 80]}
    counts, zone = analyse([person, car], zones, 100, 100)
    assert zone['potential_conflict'] and counts['person'] == 1
    assert not analyse([person], zones, 100, 100)[1]['potential_conflict']
    assert not analyse([car], zones, 100, 100)[1]['potential_conflict']


def test_invalid_zones_rejected():
    """Verify that non-normalized or degenerate polygon coordinates trigger validation errors."""
    with pytest.raises(ValueError):
        validate_zones({'pedestrian_zone': [[0, 0], [2, 0], [1, 1]]})


def test_preprocess_rgb_and_range():
    """Verify preprocessing converts BGR to RGB, resizes to (1, 3, 320, 320), and normalizes to [0, 1]."""
    x = preprocess(np.full((20, 30, 3), [0, 0, 255], dtype=np.uint8))
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
    assert x[0, 0].min() == 1 and x[0, 2].max() == 0
