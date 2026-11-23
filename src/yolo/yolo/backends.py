"""Pick how YOLO runs: TensorRT on the GPU, or OpenCV on the CPU.

backend = 'auto'      try TensorRT; if anything is missing, say why and use OpenCV
          'tensorrt'  TensorRT only (fail loudly instead of falling back)
          'opencv'    OpenCV on the CPU only

The TensorRT engine (.engine) is built from the .onnx file ON THE CAR the first
time (a few minutes, once). It is saved next to the .onnx and reused after that.
An engine only works on the GPU and TensorRT version it was built with, so it
is never copied between machines or committed to git.
"""

import os
import shutil
import subprocess

from .detector import YoloDetector

BACKENDS = ('auto', 'tensorrt', 'opencv')
TRTEXEC_CANDIDATES = ('/usr/src/tensorrt/bin/trtexec', '/usr/local/tensorrt/bin/trtexec',
                      '/opt/nvidia/tensorrt/bin/trtexec')


def default_engine_path(model_path):
    """~/models/yolov8n.onnx -> ~/models/yolov8n.engine"""
    return os.path.splitext(model_path)[0] + '.engine'


def find_trtexec():
    for path in (shutil.which('trtexec'),) + TRTEXEC_CANDIDATES:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        'trtexec (TensorRT engine builder) not found; looked on PATH and in '
        + ', '.join(TRTEXEC_CANDIDATES))


def build_engine(onnx_path, engine_path, fp16=True, trtexec=None, log=print):
    """Build a TensorRT engine from an ONNX model with trtexec. Takes minutes."""
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f'ONNX model not found: {onnx_path}')
    trtexec = trtexec or find_trtexec()
    partial = engine_path + '.partial'   # only renamed when the build succeeds
    cmd = [trtexec, f'--onnx={onnx_path}', f'--saveEngine={partial}']
    if fp16:
        cmd.append('--fp16')
    log(f'Building TensorRT engine (one time, a few minutes): {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(partial):
        tail = '\n'.join((result.stdout + result.stderr).strip().splitlines()[-15:])
        if os.path.exists(partial):
            os.remove(partial)
        raise RuntimeError(f'trtexec failed (exit {result.returncode}):\n{tail}')
    os.replace(partial, engine_path)
    log(f'Saved TensorRT engine: {engine_path}')
    return engine_path


def make_detector(backend, model_path, engine_path='', input_size=640, confidence=0.35,
                  iou=0.45, classes=('car',), auto_build=True, fp16=True,
                  names=None, info=print, warn=print):
    """Return (detector, name_of_backend_used)."""
    if backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}, got {backend!r}')
    common = dict(input_size=input_size, confidence=confidence, iou=iou, classes=classes,
                  names=names)

    if backend in ('auto', 'tensorrt'):
        try:
            if not engine_path:
                if not model_path:
                    raise ValueError('set model_path (.onnx) or engine_path (.engine)')
                engine_path = default_engine_path(model_path)
            if not os.path.isfile(engine_path):
                if not auto_build:
                    raise FileNotFoundError(f'no engine at {engine_path} and auto_build is off')
                build_engine(model_path, engine_path, fp16=fp16, log=info)
            from .trt_backend import TensorRTDetector   # needs `import tensorrt`
            detector = TensorRTDetector(engine_path, **common)
            info(f'YOLO running on the GPU with TensorRT ({engine_path})')
            return detector, 'tensorrt'
        except Exception as e:
            if backend == 'tensorrt':
                raise
            warn(f'TensorRT not used ({type(e).__name__}: {e}). '
                 'Falling back to OpenCV on the CPU (slower).')

    if not model_path:
        raise ValueError('the OpenCV backend needs model_path (.onnx)')
    detector = YoloDetector(model_path, **common)
    info(f'YOLO running on the CPU with OpenCV ({model_path})')
    return detector, 'opencv'
