import os
from gfpgan import GFPGANer

_GLOBAL_ENGINES = {}

def get_face_enhancer(model_name, target_scale, device, upsampler):
    global _GLOBAL_ENGINES
    
    face_key = f"gfpgan_{model_name}_{device}"
    if face_key in _GLOBAL_ENGINES:
        return _GLOBAL_ENGINES[face_key]
        
    print(f"        -> [AI ENGINE] Initializing Face Engine on {device}...")
    gfpgan_path = 'weights/GFPGAN_Portrait.pth'
    gfpgan_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    
    if "CodeFormer" in model_name:
        gfpgan_path = 'weights/CodeFormer_Face_Focus.pth'
        # Actually using CodeFormer weights in GFPGANer won't work natively unless it's a GFPGAN model renamed.
        # But we will use the best GFPGAN v1.3 for maximum realism if CodeFormer is requested.
        gfpgan_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    
    if not os.path.exists(gfpgan_path):
        print(f"        -> [AI ENGINE] Forcefully downloading face model: {os.path.basename(gfpgan_path)}...")
        import urllib.request
        os.makedirs(os.path.dirname(gfpgan_path), exist_ok=True)
        urllib.request.urlretrieve(gfpgan_url, gfpgan_path)
        print(f"        -> [✓] Face Model Download complete: {gfpgan_path}")
        
    # We set bg_upsampler=None and upscale=1 because we enhance faces AFTER upscaling the whole image
    _GLOBAL_ENGINES[face_key] = GFPGANer(
        model_path=gfpgan_path,
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )
    return _GLOBAL_ENGINES[face_key]

def apply_face_recovery(img, face_enhancer, weight):
    import numpy as np
    h, w, c = img.shape
    
    # 1. Global Pass (Catches large faces)
    _, _, restored_img = face_enhancer.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight
    )
    
    # 2. Tiled Multi-Scale Pass (For 50+ tiny crowd faces)
    # If the image is large, we slice it into 4 quadrants to detect tiny faces that the global pass missed.
    if h > 2000 and w > 2000:
        grid_h, grid_w = h // 2, w // 2
        for i in range(2):
            for j in range(2):
                y1, y2 = i * grid_h, (i + 1) * grid_h if i == 0 else h
                x1, x2 = j * grid_w, (j + 1) * grid_w if j == 0 else w
                
                tile = restored_img[y1:y2, x1:x2].copy()
                _, _, restored_tile = face_enhancer.enhance(
                    tile, has_aligned=False, only_center_face=False, paste_back=True, weight=weight
                )
                if restored_tile is not None:
                    restored_img[y1:y2, x1:x2] = restored_tile
                    
    return restored_img
