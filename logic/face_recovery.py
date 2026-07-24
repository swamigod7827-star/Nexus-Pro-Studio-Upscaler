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
    if "CodeFormer" in model_name:
        gfpgan_path = 'weights/CodeFormer_Face_Focus.pth'
    
    if not os.path.exists(gfpgan_path):
        print(f"        -> [AI ENGINE] Forcefully downloading face model: {os.path.basename(gfpgan_path)}...")
        import urllib.request
        gfpgan_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        os.makedirs(os.path.dirname(gfpgan_path), exist_ok=True)
        urllib.request.urlretrieve(gfpgan_url, gfpgan_path)
        print(f"        -> [✓] Face Model Download complete: {gfpgan_path}")
        
    _GLOBAL_ENGINES[face_key] = GFPGANer(
        model_path=gfpgan_path,
        upscale=target_scale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler,
        device=device
    )
    return _GLOBAL_ENGINES[face_key]

def apply_face_recovery(img, face_enhancer, weight):
    _, _, restored_img = face_enhancer.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight
    )
    return restored_img
