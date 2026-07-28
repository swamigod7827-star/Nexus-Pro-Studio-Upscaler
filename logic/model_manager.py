import os
import glob

# --- AUTOMATIC BASICSR PATCH ---
# Fixes torchvision.transforms.functional_tensor deprecation in newer PyTorch
try:
    import site
    paths = getattr(site, 'getsitepackages', lambda: [])()
    if hasattr(site, 'getusersitepackages'):
        paths.append(site.getusersitepackages())
    for p in paths:
        for f in glob.glob(os.path.join(p, "basicsr", "data", "degradations.py")):
            with open(f, 'r') as file:
                content = file.read()
            if 'functional_tensor' in content:
                content = content.replace('functional_tensor', 'functional')
                with open(f, 'w') as file:
                    file.write(content)
except Exception:
    pass
# -------------------------------

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
import numpy as np

class SpandrelUpscaler:
    def __init__(self, model_path, device, half_precision):
        from spandrel import ModelLoader
        self.device = torch.device(device)
        self.half = half_precision
        self.model = ModelLoader().load_from_file(model_path).eval().to(self.device)
        if self.half:
            self.model = self.model.half()
        self.scale = getattr(self.model, 'scale', 4)

    @torch.no_grad()
    def enhance(self, img, outscale=None):
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if self.half:
            img_t = img_t.half()
        img_t = img_t.to(self.device)

        # Simple Tiling to avoid OOM
        b, c, h, w = img_t.shape
        tile = 400
        pad = 10
        scale = self.scale
        out_t = torch.zeros((b, c, h * scale, w * scale), device=self.device if self.half else 'cpu')
        if self.half: out_t = out_t.half()

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                in_y1, in_y2 = max(0, y - pad), min(h, y + tile + pad)
                in_x1, in_x2 = max(0, x - pad), min(w, x + tile + pad)
                
                in_tile = img_t[:, :, in_y1:in_y2, in_x1:in_x2]
                out_tile = self.model(in_tile)

                out_y1, out_x1 = (y - in_y1) * scale, (x - in_x1) * scale
                out_y2, out_x2 = out_y1 + min(tile, h - y) * scale, out_x1 + min(tile, w - x) * scale

                in_y1_out, in_y2_out = y * scale, min(h, y + tile) * scale
                in_x1_out, in_x2_out = x * scale, min(w, x + tile) * scale

                out_t[:, :, in_y1_out:in_y2_out, in_x1_out:in_x2_out] = out_tile[:, :, out_y1:out_y2, out_x1:out_x2]

        out_img = out_t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
        
        # If outscale is explicitly different from the model scale, we'd resize here, but usually it matches
        import cv2
        if outscale is not None and outscale != self.scale:
            out_img = cv2.resize(out_img, (w * outscale, h * outscale), interpolation=cv2.INTER_LANCZOS4)
            
        return out_img, None

_GLOBAL_ENGINES = {}

def get_upsampler(model_name, device, half_precision):
    global _GLOBAL_ENGINES
    
    cache_key = f"{model_name}_{device}_{half_precision}"
    upsampler_key = f"upsampler_{cache_key}"
    
    if upsampler_key in _GLOBAL_ENGINES:
        return _GLOBAL_ENGINES[upsampler_key]
        
    use_spandrel = False
    model_path = 'weights/RealESRGAN_v4_General.pth'
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    if "Anime" in model_name:
        model_path = 'weights/Anime_Sharp_Illustrations.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    elif "SwinIR" in model_name:
        model_path = 'weights/SwinIR_Texture_Detail.pth'
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
        use_spandrel = True
    elif "BSRGAN" in model_name:
        model_path = 'weights/BSRGAN_Real_World.pth'
        url = 'https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth'
        use_spandrel = True
    elif "HAT" in model_name:
        model_path = 'weights/HAT_High_Accuracy.pth'
        url = 'https://github.com/XPixelGroup/HAT/releases/download/v1.0/Real_HAT_GAN_SRx4.pth'
        use_spandrel = True
    elif "NAFNet" in model_name:
        model_path = 'weights/NAFNet_Fast_Restoration.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
        use_spandrel = True

    if not os.path.exists(model_path):
        print(f"        -> [AI ENGINE] Forcefully downloading model weights for {model_name}...", flush=True)
        import urllib.request
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
        print(f"        -> [✓] Download complete: {model_path}")

    print(f"        -> [AI ENGINE] Initializing {model_name} on {device}...")
    
    if use_spandrel:
        _GLOBAL_ENGINES[upsampler_key] = SpandrelUpscaler(model_path, device, half_precision)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6 if "Anime" in model_name else 23, num_grow_ch=32, scale=4)
        _GLOBAL_ENGINES[upsampler_key] = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=half_precision,
            device=device
        )
    
    return _GLOBAL_ENGINES[upsampler_key]
