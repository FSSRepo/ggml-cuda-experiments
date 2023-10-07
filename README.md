# ggml-cuda-experiments
CUDA Experimients

## Build ggml test you need Cmake

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Build cuda test you need Nvidia Dev Toolkit

```bash
nvcc conv2d.cu -o conv2d-cublas
conv2d-cublas
```