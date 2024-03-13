nvcc src/flash-matrix.cu -o build/tensor -lineinfo -arch sm_70 && build\tensor
REM nvcc src/misc/coalescing.cu -o build/coalescing -lineinfo -arch sm_70 && build\coalescing
REM nvcc src/misc/cudaTensorCoreGemm.cu -o build/tensor_gemm -lineinfo -arch sm_70 && build\tensor_gemm
REM nvcc src/misc/simpleCooperativeGroups.cu -o build/cta-cuda -lineinfo -arch sm_70 && build\cta-cuda
REM nvcc src/misc/shared-memory.cu -o build/shmem -lineinfo -arch sm_70 && build\shmem
REM nvcc src/misc/transpose.cu -o build/transpose -lineinfo -arch sm_70 && build\transpose