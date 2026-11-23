"""Tests for the ROS-free detection logic (no model or camera needed)."""

import numpy as np

from yolo.detector import (class_ids_for, color_mask, draw_detections, orange_mask,
                           parse_yolo_output)


def fake_output(guesses, num_classes=80, num_boxes=8400):
    """Build a fake YOLOv8 output (1, 4+classes, boxes) from (cx,cy,w,h,class,score)."""
    out = np.zeros((1, 4 + num_classes, num_boxes), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(guesses):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out


def test_no_detections_on_empty_output():
    assert parse_yolo_output(fake_output([]), 640, 480) == []


def test_low_confidence_is_dropped():
    out = fake_output([(320, 320, 100, 100, 2, 0.2)])
    assert parse_yolo_output(out, 640, 640, confidence=0.35) == []


def test_box_is_scaled_to_frame_size():
    # A box at the centre of the 640x640 input, frame is 1280x720.
    out = fake_output([(320, 320, 64, 64, 2, 0.9)])
    [d] = parse_yolo_output(out, 1280, 720)
    assert d['class_name'] == 'car'
    assert d['confidence'] == 0.9
    assert d['bbox_xyxy_pixels'] == [576.0, 324.0, 704.0, 396.0]
    assert d['center_xy_pixels'] == [640.0, 360.0]


def test_nms_keeps_one_of_two_overlapping_boxes():
    out = fake_output([(320, 320, 100, 100, 2, 0.9),
                       (322, 321, 100, 100, 2, 0.8)])
    dets = parse_yolo_output(out, 640, 640)
    assert len(dets) == 1
    assert dets[0]['confidence'] == 0.9


def test_separate_objects_are_both_kept():
    out = fake_output([(100, 100, 50, 50, 2, 0.9), (500, 500, 50, 50, 0, 0.7)])
    names = {d['class_name'] for d in parse_yolo_output(out, 640, 640)}
    assert names == {'car', 'person'}


def test_orange_mask_finds_orange_only():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, :10] = (0, 128, 255)   # orange in BGR
    frame[:, 10:] = (255, 0, 0)     # blue
    mask = orange_mask(frame)
    assert mask[:, :10].all()
    assert not mask[:, 10:].any()


def test_draw_does_not_change_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = {'class_name': 'car', 'confidence': 0.9,
           'bbox_xyxy_pixels': [10, 10, 50, 50]}
    out = draw_detections(frame, [det])
    assert not frame.any()
    assert out.any()


def test_class_filter_keeps_only_cars():
    out = fake_output([(100, 100, 50, 50, 2, 0.9), (500, 500, 50, 50, 0, 0.9)])
    dets = parse_yolo_output(out, 640, 640, keep_class_ids={2})
    assert [d['class_name'] for d in dets] == ['car']


def test_class_names_to_ids():
    assert class_ids_for(['car']) == {2}
    assert class_ids_for(['all']) is None
    assert class_ids_for([]) is None


def test_unknown_class_name_is_an_error():
    try:
        class_ids_for(['racecar'])
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_yellow_and_orange_are_separate():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, :10] = (0, 128, 255)   # orange in BGR
    frame[:, 10:] = (0, 255, 255)   # yellow in BGR
    orange = color_mask(frame, 'orange')
    yellow = color_mask(frame, 'yellow')
    assert orange[:, :10].all() and not orange[:, 10:].any()
    assert yellow[:, 10:].all() and not yellow[:, :10].any()


def test_skin_tone_is_not_a_barrier():
    frame = np.full((10, 10, 3), (120, 160, 220), dtype=np.uint8)   # light skin, BGR
    assert not color_mask(frame, 'orange').any()
    assert not color_mask(frame, 'yellow').any()


def test_bad_colour_name_is_an_error():
    try:
        color_mask(np.zeros((2, 2, 3), dtype=np.uint8), 'purple')
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_custom_model_names():
    names = ['racerbot_car', 'barrier']
    out = fake_output([(100, 100, 50, 50, 0, 0.9), (500, 500, 50, 50, 1, 0.8)], num_classes=2)
    dets = parse_yolo_output(out, 640, 640, names=names,
                             keep_class_ids=class_ids_for(['barrier'], names))
    assert [d['class_name'] for d in dets] == ['barrier']
