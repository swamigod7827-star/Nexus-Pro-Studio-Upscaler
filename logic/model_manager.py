import os
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# --- AUTOMATIC BASICSR PATCH ---
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

_GLOBAL_ENGINES = {}

def get_upsampler(model_name, device, half_precision):
    global _GLOBAL_ENGINES
    
    cache_key = f"{model_name}_{device}_{half_precision}"
    upsampler_key = f"upsampler_{cache_key}"
    
    if upsampler_key in _GLOBAL_ENGINES:
        return _GLOBAL_ENGINES[upsampler_key]
        
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = 'weights/RealESRGAN_v4_General.pth'
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    if "Anime" in model_name:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        model_path = 'weights/Anime_Sharp_Illustrations.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    elif "SwinIR" in model_name:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/SwinIR_Texture_Detail.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
    elif "BSRGAN" in model_name:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/BSRGAN_Real_World.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
    elif "HAT" in model_name:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/HAT_High_Accuracy.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    elif "NAFNet" in model_name:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/NAFNet_Fast_Restoration.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
    elif "TextMaster" in model_name:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/TextMaster_V1.pth'
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'

    if not os.path.exists(model_path):
        print(f"        -> [AI ENGINE] Forcefully downloading model weights for {model_name}...", flush=True)
        import urllib.request
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
        print(f"        -> [✓] Download complete: {model_path}")

    print(f"        -> [AI ENGINE] Initializing RealESRGANer upsampler on {device}...")
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
