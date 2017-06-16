import pyopencl
import tectosaur.util.gpu as gpu

def get_gpu_platform():
    cuda_platform_idx = None
    platforms = pyopencl.get_platforms()
    for i, p in enumerate(platforms):
        if 'CUDA' in p.name:
            cuda_platform_idx = i
            break
    if cuda_platform_idx is None:
        raise Exception("No CUDA platform.")
    cuda_platform = platforms[cuda_platform_idx]
    return cuda_platform

def how_many_gpus():
    return len(get_gpu_platform().get_devices())

def use_gpu(idx):
    ctx = pyopencl.Context(devices = [get_gpu_platform().get_devices()[idx]])
    gpu.initialize_with_ctx(ctx)
