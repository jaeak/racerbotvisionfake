"""Tests for picking the backend and building the engine (no GPU needed)."""

import os
import stat

import numpy as np

from yolo import backends
from yolo.detector import preprocess


def fake_trtexec(tmp_path, exit_code=0, write_engine=True):
    """A stand-in trtexec that writes the --saveEngine file (or fails)."""
    script = tmp_path / 'trtexec'
    body = '#!/bin/sh\n'
    if write_engine:
        body += ('for a in "$@"; do case "$a" in --saveEngine=*) '
                 'echo engine > "${a#--saveEngine=}";; esac; done\n')
    body += f'echo "fake trtexec output"\nexit {exit_code}\n'
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_default_engine_path():
    assert backends.default_engine_path('/m/yolov8n.onnx') == '/m/yolov8n.engine'


def test_build_engine_saves_file(tmp_path):
    onnx = tmp_path / 'm.onnx'
    onnx.write_text('x')
    engine = str(tmp_path / 'm.engine')
    backends.build_engine(str(onnx), engine, trtexec=fake_trtexec(tmp_path), log=lambda m: None)
    assert os.path.isfile(engine)
    assert not os.path.exists(engine + '.partial')


def test_failed_build_leaves_no_engine(tmp_path):
    onnx = tmp_path / 'm.onnx'
    onnx.write_text('x')
    engine = str(tmp_path / 'm.engine')
    try:
        backends.build_engine(str(onnx), engine, trtexec=fake_trtexec(tmp_path, exit_code=1),
                              log=lambda m: None)
    except RuntimeError as e:
        assert 'fake trtexec output' in str(e)
    else:
        raise AssertionError('expected RuntimeError')
    assert not os.path.exists(engine)
    assert not os.path.exists(engine + '.partial')


class FakeCpu:
    def __init__(self, model_path, **kw):
        self.model_path = model_path


def test_auto_falls_back_to_opencv(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, 'YoloDetector', FakeCpu)
    warnings = []
    # No engine and no trtexec available -> fall back.
    monkeypatch.setattr(backends, 'find_trtexec',
                        lambda: (_ for _ in ()).throw(FileNotFoundError('no trtexec')))
    onnx = tmp_path / 'm.onnx'
    onnx.write_text('x')
    det, used = backends.make_detector('auto', str(onnx), info=lambda m: None,
                                       warn=warnings.append)
    assert used == 'opencv' and isinstance(det, FakeCpu)
    assert 'no trtexec' in warnings[0]


def test_tensorrt_only_does_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, 'YoloDetector', FakeCpu)
    monkeypatch.setattr(backends, 'find_trtexec',
                        lambda: (_ for _ in ()).throw(FileNotFoundError('no trtexec')))
    onnx = tmp_path / 'm.onnx'
    onnx.write_text('x')
    try:
        backends.make_detector('tensorrt', str(onnx), info=lambda m: None, warn=lambda m: None)
    except FileNotFoundError:
        return
    raise AssertionError('expected the error, not a fallback')


def test_opencv_backend_skips_tensorrt(monkeypatch):
    monkeypatch.setattr(backends, 'YoloDetector', FakeCpu)
    det, used = backends.make_detector('opencv', '/m/x.onnx', info=lambda m: None)
    assert used == 'opencv' and det.model_path == '/m/x.onnx'


def test_bad_backend_name():
    try:
        backends.make_detector('cuda', '/m/x.onnx')
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_preprocess_shape_and_range():
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
    blob = preprocess(frame, 640)
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert abs(float(blob.max()) - 1.0) < 1e-6
