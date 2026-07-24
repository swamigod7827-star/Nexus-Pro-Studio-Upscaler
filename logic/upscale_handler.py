import os
import uuid
import tempfile
import numpy as np

# Deep Analysis: True 64K Processing via Disk-Streaming (Memmap)
# Bypasses 12GB RAM limit of Colab/Local by writing huge arrays to SSD.
orig_zeros = np.zeros

def memmap_zeros(shape, dtype=float, order='C', *, like=None):
    if isinstance(shape, (tuple, list)) and len(shape) == 3:
        try:
            bytes_req = np.prod(shape) * np.dtype(dtype).itemsize
            # If requested array > 1GB, swap to disk!
            if bytes_req > 1024 * 1024 * 1024:
                temp_file = os.path.join(tempfile.gettempdir(), f"memmap_{uuid.uuid4().hex}.dat")
                print(f"🔥 SWAPPING TO DISK: {bytes_req / (1024**3):.2f} GB Array -> {temp_file}")
                return np.memmap(temp_file, dtype=dtype, mode='w+', shape=tuple(shape))
        except Exception:
            pass
    # For `like=like` kwargs passing compatibility in newer numpy
    if like is not None:
        return orig_zeros(shape, dtype=dtype, order=order, like=like)
    return orig_zeros(shape, dtype=dtype, order=order)

def perform_upscale(img, upsampler, target_scale):
    # Hijack np.zeros for RealESRGAN
    np.zeros = memmap_zeros
    try:
        upscaled, _ = upsampler.enhance(img, outscale=target_scale)
    finally:
        # Restore np.zeros
        np.zeros = orig_zeros
    return upscaled
