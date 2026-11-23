"""Run a YOLOv8 TensorRT engine on the Jetson's GPU.

Same input and output as YoloDetector in detector.py (picture in, list of
detection dicts out); only the part that runs the model is different.

The car has TensorRT 10 but no PyCUDA / CuPy, so GPU memory is handled by
calling CUDA's own runtime library (libcudart, part of the CUDA install)
through ctypes. Nothing extra needs installing besides TensorRT's Python
bindings (`import tensorrt`), which ship with JetPack.
"""

import ctypes
import ctypes.util

import numpy as np
import tensorrt as trt

from .detector import class_ids_for, COCO_NAMES, parse_yolo_output, preprocess

_HOST_TO_DEVICE = 1    # cudaMemcpyHostToDevice
_DEVICE_TO_HOST = 2    # cudaMemcpyDeviceToHost


class CudaRuntime:
    """The few CUDA runtime calls we need, loaded straight from libcudart."""

    CANDIDATES = ('libcudart.so', 'libcudart.so.13', 'libcudart.so.12',
                  '/usr/local/cuda/lib64/libcudart.so',
                  '/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so')

    def __init__(self):
        errors = []
        self.lib = None
        for name in (ctypes.util.find_library('cudart'),) + self.CANDIDATES:
            if not name:
                continue
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError as e:
                errors.append(f'{name}: {e}')
        if self.lib is None:
            raise OSError('could not load the CUDA runtime (libcudart): ' + '; '.join(errors))

        lib = self.lib
        void_p, size_t, c_int = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        lib.cudaMalloc.argtypes = [ctypes.POINTER(void_p), size_t]
        lib.cudaFree.argtypes = [void_p]
        lib.cudaMemcpyAsync.argtypes = [void_p, void_p, size_t, c_int, void_p]
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(void_p)]
        lib.cudaStreamSynchronize.argtypes = [void_p]
        lib.cudaStreamDestroy.argtypes = [void_p]
        for fn in ('cudaMalloc', 'cudaFree', 'cudaMemcpyAsync', 'cudaStreamCreate',
                   'cudaStreamSynchronize', 'cudaStreamDestroy'):
            getattr(lib, fn).restype = c_int
        lib.cudaGetErrorString.argtypes = [c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, code, what):
        if code != 0:
            raise RuntimeError(f'{what} failed: {self.lib.cudaGetErrorString(code).decode()}')

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), 'cudaMalloc')
        return ptr

    def free(self, ptr):
        self._check(self.lib.cudaFree(ptr), 'cudaFree')

    def stream_create(self):
        stream = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)), 'cudaStreamCreate')
        return stream

    def stream_destroy(self, stream):
        self._check(self.lib.cudaStreamDestroy(stream), 'cudaStreamDestroy')

    def to_device(self, device_ptr, host_array, stream):
        self._check(self.lib.cudaMemcpyAsync(
            device_ptr, host_array.ctypes.data_as(ctypes.c_void_p), host_array.nbytes,
            _HOST_TO_DEVICE, stream), 'cudaMemcpyAsync (to GPU)')

    def to_host(self, host_array, device_ptr, stream):
        self._check(self.lib.cudaMemcpyAsync(
            host_array.ctypes.data_as(ctypes.c_void_p), device_ptr, host_array.nbytes,
            _DEVICE_TO_HOST, stream), 'cudaMemcpyAsync (from GPU)')

    def synchronize(self, stream):
        self._check(self.lib.cudaStreamSynchronize(stream), 'cudaStreamSynchronize')


class TensorRTDetector:
    """Loads a TensorRT .engine once, then finds objects in frames on the GPU."""

    def __init__(self, engine_path, input_size=640, confidence=0.35, iou=0.45,
                 classes=('car',), names=None):
        self.cuda = CudaRuntime()
        self.confidence = confidence
        self.iou = iou
        self.names = list(names) if names else COCO_NAMES
        self.keep_class_ids = class_ids_for(list(classes), self.names)
        self._buffers = []
        self.stream = None

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f'could not load {engine_path}. It was probably built with a different '
                'TensorRT version or on another machine: delete it and let it rebuild.')
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f'expected 1 input and 1 output, got {inputs} / {outputs}')

        # Input shape: fixed in a normal YOLOv8 export (1, 3, 640, 640).
        in_shape = tuple(self.engine.get_tensor_shape(inputs[0]))
        if -1 in in_shape:  # dynamic export: pick batch 1 at our input size
            in_shape = (1, 3, input_size, input_size)
            self.context.set_input_shape(inputs[0], in_shape)
        self.input_size = int(in_shape[2])   # the engine decides, not the setting
        out_shape = tuple(self.context.get_tensor_shape(outputs[0]))

        self.stream = self.cuda.stream_create()
        self.in_host, self.in_dev = self._allocate(inputs[0], in_shape)
        self.out_host, self.out_dev = self._allocate(outputs[0], out_shape)

    def _allocate(self, name, shape):
        host = np.empty(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
        device = self.cuda.malloc(host.nbytes)
        self._buffers.append(device)
        self.context.set_tensor_address(name, device.value)
        return host, device

    def infer_raw(self, blob):
        """Run the network on a preprocessed blob, return its raw output."""
        np.copyto(self.in_host, blob, casting='same_kind')
        self.cuda.to_device(self.in_dev, self.in_host, self.stream)
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError('TensorRT execute_async_v3 failed')
        self.cuda.to_host(self.out_host, self.out_dev, self.stream)
        self.cuda.synchronize(self.stream)
        return self.out_host.astype(np.float32, copy=False)

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        output = self.infer_raw(preprocess(frame_bgr, self.input_size))
        return parse_yolo_output(output, w, h, self.input_size, self.confidence, self.iou,
                                 names=self.names, keep_class_ids=self.keep_class_ids)

    def close(self):
        for ptr in self._buffers:
            self.cuda.free(ptr)
        self._buffers = []
        if self.stream is not None:
            self.cuda.stream_destroy(self.stream)
            self.stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
