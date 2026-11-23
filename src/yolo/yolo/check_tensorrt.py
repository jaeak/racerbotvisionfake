"""Check TensorRT works on this machine, build the engine, and compare speeds.

    ros2 run yolo yolo_check --model ~/models/yolov8n.onnx

Steps (each prints OK or what's wrong):
  1. TensorRT Python bindings import
  2. CUDA runtime (libcudart) loads
  3. trtexec found
  4. engine exists, or gets built from the .onnx (a few minutes, once)
  5. engine runs on the GPU
  6. TensorRT output matches OpenCV's (same model, so they should agree)
  7. speed: TensorRT (GPU) vs OpenCV (CPU)

No ROS and no camera needed.
"""

import argparse
import os
import time

import numpy as np


def ok(msg):
    print(f'  OK   {msg}')


def fail(msg, fix=''):
    print(f'  FAIL {msg}')
    if fix:
        print(f'       fix: {fix}')
    raise SystemExit(1)


def timed(fn, runs):
    fn()                                   # warm-up (first run is always slow)
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs * 1000


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='YOLOv8 .onnx file')
    p.add_argument('--engine', default='', help='engine path (default: next to the .onnx)')
    p.add_argument('--runs', type=int, default=30)
    p.add_argument('--no-fp16', action='store_true', help='build a full-precision engine')
    args = p.parse_args()
    model = os.path.expanduser(args.model)

    print('1. TensorRT Python bindings')
    try:
        import tensorrt as trt
        ok(f'tensorrt {trt.__version__}')
    except ImportError as e:
        fail(f'import tensorrt: {e}',
             "TensorRT's Python package is missing. Ask whoever manages the car about "
             'installing python3-libnvinfer (JetPack TensorRT Python bindings).')

    print('2. CUDA runtime')
    from yolo.trt_backend import CudaRuntime
    try:
        cuda = CudaRuntime()
        ok(f'loaded {cuda.lib._name}')
    except OSError as e:
        fail(str(e), 'check CUDA is installed: ls /usr/local/cuda/lib64/libcudart*')

    print('3. trtexec')
    from yolo.backends import build_engine, default_engine_path, find_trtexec
    try:
        ok(find_trtexec())
    except FileNotFoundError as e:
        fail(str(e), 'find it with: find / -name trtexec 2>/dev/null')

    print('4. engine')
    engine = os.path.expanduser(args.engine) or default_engine_path(model)
    if os.path.isfile(engine):
        ok(f'already built: {engine}')
    else:
        start = time.time()
        try:
            build_engine(model, engine, fp16=not args.no_fp16, log=lambda m: print(f'       {m}'))
        except Exception as e:
            fail(str(e))
        ok(f'built in {time.time() - start:.0f} s: {engine}')

    print('5. run on the GPU')
    from yolo.detector import preprocess, YoloDetector
    from yolo.trt_backend import TensorRTDetector
    try:
        trt_det = TensorRTDetector(engine, classes=['all'])
    except Exception as e:
        fail(str(e), f'delete {engine} and run this again to rebuild it')
    frame = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
    blob = preprocess(frame, trt_det.input_size)
    trt_out = trt_det.infer_raw(blob).copy()
    ok(f'input {trt_det.in_host.shape} {trt_det.in_host.dtype}, '
       f'output {trt_out.shape} {trt_out.dtype}')

    print('6. compare with OpenCV (same model)')
    cv_det = None
    try:
        cv_det = YoloDetector(model, input_size=trt_det.input_size, classes=['all'])
        cv_det.net.setInput(blob)
        cv_out = cv_det.net.forward()
    except Exception as e:
        print(f'  SKIP OpenCV could not run this model ({type(e).__name__}: {e}).')
        print('       Not a problem for TensorRT; it only means no CPU fallback.')
    if cv_det is not None:
        if cv_out.shape != trt_out.shape:
            fail(f'output shapes differ: OpenCV {cv_out.shape} vs TensorRT {trt_out.shape}')
        box_diff = float(np.max(np.abs(cv_out[:, :4] - trt_out[:, :4])))
        score_diff = float(np.max(np.abs(cv_out[:, 4:] - trt_out[:, 4:])))
        msg = f'max difference: boxes {box_diff:.2f} px, scores {score_diff:.3f}'
        if box_diff < 5 and score_diff < 0.05:
            ok(msg + ' (FP16 rounding only)')
        else:
            print(f'  WARN {msg} - larger than expected, check detections look the same')

    print('7. speed (one 1280x720 frame, network + post-processing)')
    trt_ms = timed(lambda: trt_det.detect(frame), args.runs)
    ok(f'TensorRT (GPU): {trt_ms:6.1f} ms  = {1000 / trt_ms:5.1f} FPS')
    if cv_det is not None:
        cv_ms = timed(lambda: cv_det.detect(frame), max(3, args.runs // 5))
        ok(f'OpenCV   (CPU): {cv_ms:6.1f} ms  = {1000 / cv_ms:5.1f} FPS')
        ok(f'TensorRT is {cv_ms / trt_ms:.1f}x faster')
    trt_det.close()
    print('\nAll good: the YOLO node will use TensorRT (backend: auto).')


if __name__ == '__main__':
    main()
