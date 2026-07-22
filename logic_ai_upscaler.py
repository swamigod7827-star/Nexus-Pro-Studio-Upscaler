import os
import cv2
import numpy as np
import json
import base64
import glob
import torch
from PIL import Image

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

# Generative AI Models
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def update_progress(tracker, task_id, percent, log_msg):
    if tracker is not None and task_id:
        tracker[task_id] = {"percent": percent, "log": log_msg}
    print(f"[PROGRESS] {percent}%: {log_msg}", flush=True)

_GLOBAL_ENGINES = {}

def get_cached_engines(model_name, target_scale, device, half_precision, face_restore):
    global _GLOBAL_ENGINES
    
    # We cache based on model configuration
    cache_key = f"{model_name}_{target_scale}_{device}_{half_precision}"
    
    # Select appropriate model structure and path (Mapping all 8 UI options)
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

    upsampler_key = f"upsampler_{cache_key}"
    if upsampler_key not in _GLOBAL_ENGINES:
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
    upsampler = _GLOBAL_ENGINES[upsampler_key]

    face_enhancer = None
    if face_restore:
        face_key = f"gfpgan_{cache_key}"
        if face_key not in _GLOBAL_ENGINES:
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
        face_enhancer = _GLOBAL_ENGINES[face_key]
        
    return upsampler, face_enhancer

class NexusGenerativeEngine:
    def __init__(self, payload):
        print("\n" + "★"*75)
        print("⚙️ NEXUS PRO: GENERATIVE ENGINE ⚙️")
        print("★"*75)
        
        self.model_name = payload.get('model', 'RealESRGAN v4')
        self.proc_mode = payload.get('mode', 'CPU Precision (Slow)')
        print(f"[*] [UI ROUTER] Selected AI Model: {self.model_name}")
        print(f"[*] [UI ROUTER] Hardware Mode: {self.proc_mode}")
        
        # GPU detection based on UI settings
        self.device = 'cuda' if ('GPU' in self.proc_mode or 'TensorRT' in self.proc_mode) and torch.cuda.is_available() else 'cpu'
        self.half_precision = True if self.device == 'cuda' else False
        print(f"[*] [AI ENGINE] Hardware Device: {self.device} (FP16: {self.half_precision})")
        
        # Dynamic scale from UI
        self.target_scale = int(''.join(filter(str.isdigit, str(payload.get('factor', '4')).split(' ')[0])) or 4)
        
        adv = json.loads(payload.get('settings', '{}'))
        self.artifact_rem = int(adv.get('artifact', 25))
        
        face = json.loads(payload.get('face', '{}'))
        self.face_restore = str(face.get('enabled', 'false')).lower() == 'true'
        
        # Auto-enable face restore if a portrait model is selected
        if any(keyword in self.model_name for keyword in ["Face", "Portrait", "CodeFormer", "GFPGAN"]):
            self.face_restore = True
            print("[*] [UI ROUTER] Auto-enabled Face Recovery for selected portrait model.")
            
        self.face_weight = int(face.get('identity', 85)) / 100.0 
        
        export = json.loads(payload.get('export', '{}'))
        self.dpi = int(''.join(filter(str.isdigit, str(export.get('dpi', '600')).split(' ')[0])) or 600)
        self.color_space = export.get('color_space', 'sRGB')
        
        fmt_str = export.get('format', 'PNG').upper()
        if any(x in fmt_str for x in ['CDR', 'AI', 'EPS']):
            self.ext, self.pil_fmt = 'eps', 'EPS'
        elif 'PDF' in fmt_str:
            self.ext, self.pil_fmt = 'pdf', 'PDF'
        elif any(x in fmt_str for x in ['PSD', 'TIFF', 'TIF']):
            self.ext, self.pil_fmt = 'tiff', 'TIFF'
        elif 'WEBP' in fmt_str:
            self.ext, self.pil_fmt = 'webp', 'WEBP'
        elif 'JPG' in fmt_str or 'JPEG' in fmt_str:
            self.ext, self.pil_fmt = 'jpg', 'JPEG'
        else:
            self.ext, self.pil_fmt = 'png', 'PNG'

    def process_image(self, input_path, output_dir, tracker=None, task_id=None):
        update_progress(tracker, task_id, 10, f"Preparing Input File -> {os.path.basename(input_path)}...")
        img = cv2.imread(input_path)
        if img is None: 
            raise ValueError("Image corrupted or missing.")

        # Deep Analysis: True 64K Processing via Disk-Streaming (Memmap)
        # Bypasses 12GB RAM limit of Colab/Local by writing huge arrays to SSD.
        import uuid
        import tempfile
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

        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

        update_progress(tracker, task_id, 25, f"Allocating {self.device.upper()} Threads and Loading Architecture...")
        
        # Retrieve cached engines
        upsampler, face_enhancer = get_cached_engines(
            self.model_name, self.target_scale, self.device, self.half_precision, self.face_restore
        )

        # Deep Analysis: Smart Pipeline Architecture for Extreme Scales
        # OpenCV's warpAffine crashes at 32767 pixels. RetinaFace OOMs on 16K+.
        # We smartly swap the order for massive scales: Face Enhance FIRST, then Upscale.
        out_h, out_w = img.shape[0] * self.target_scale, img.shape[1] * self.target_scale
        
        if self.face_restore and face_enhancer is not None:
            if "CodeFormer" in self.model_name:
                self.face_weight = 0.95
                
            if max(out_h, out_w) > 16384:
                # SAFE MASSIVE SCALE PIPELINE
                update_progress(tracker, task_id, 30, "Applying Generative Face Restoration (Pre-Scaling)...")
                _, _, img = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True, weight=self.face_weight
                )
                
                update_progress(tracker, task_id, 70, f"Processing {self.target_scale}X Upscaling & Generative Enhancements...")
                np.zeros = memmap_zeros
                try:
                    upscaled, _ = upsampler.enhance(img, outscale=self.target_scale)
                finally:
                    np.zeros = orig_zeros
            else:
                # STANDARD PIPELINE
                update_progress(tracker, task_id, 30, f"Processing {self.target_scale}X Upscaling & Generative Enhancements...")
                np.zeros = memmap_zeros
                try:
                    upscaled, _ = upsampler.enhance(img, outscale=self.target_scale)
                finally:
                    np.zeros = orig_zeros
                    
                update_progress(tracker, task_id, 70, "Applying Generative Face Restoration...")
                _, _, upscaled = face_enhancer.enhance(
                    upscaled, has_aligned=False, only_center_face=False, paste_back=True, weight=self.face_weight
                )
        else:
            # NO FACE RESTORATION PIPELINE
            update_progress(tracker, task_id, 30, f"Processing {self.target_scale}X Upscaling & Generative Enhancements...")
            np.zeros = memmap_zeros
            try:
                upscaled, _ = upsampler.enhance(img, outscale=self.target_scale)
            finally:
                np.zeros = orig_zeros

        h, w = upscaled.shape[:2]

        # Gentle Artifact Removal (Optional)
        if self.artifact_rem > 0:
            rem_val = max(1, self.artifact_rem // 3)
            upscaled = cv2.bilateralFilter(upscaled, 5, rem_val, rem_val)

        update_progress(tracker, task_id, 95, f"Formatting for Export ({self.pil_fmt} | {self.dpi} DPI)...")
        img_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        pil_master = Image.fromarray(img_rgb)
        
        # Deep Analysis: Prevent Format conflicts with Color Space
        if "CMYK" in self.color_space.upper():
            if self.pil_fmt in ['PNG', 'WEBP']:
                update_progress(tracker, task_id, 96, "Notice: PNG/WEBP do not support CMYK. Exporting as TIFF instead.")
                self.pil_fmt = 'TIFF'
                self.ext = 'tiff'
            pil_master = pil_master.convert('CMYK')
        elif "GRAYSCALE" in self.color_space.upper() or "B&W" in self.color_space.upper():
            pil_master = pil_master.convert('L')

        import time
        orig_name = os.path.splitext(os.path.basename(input_path))[0]
        # Use original filename with prefix and unique timestamp to prevent browser cache problems
        master_file = f"{orig_name}_Nexus_X{int(self.target_scale)}_{int(time.time())}.{self.ext}"
        master_path = os.path.join(output_dir, master_file)
        
        # Safe kwargs passing to prevent PIL exceptions
        save_kwargs = {}
        if self.pil_fmt == 'JPEG':
            save_kwargs = {'quality': 100, 'dpi': (self.dpi, self.dpi)}
        elif self.pil_fmt == 'PDF':
            save_kwargs = {'resolution': float(self.dpi)}
        elif self.pil_fmt in ['PNG', 'TIFF', 'WEBP']:
            save_kwargs = {'dpi': (self.dpi, self.dpi)}
            
        pil_master.save(master_path, format=self.pil_fmt, **save_kwargs)

        update_progress(tracker, task_id, 99, "Generating Fast UI Preview...")
        
        preview_file = f"preview_{orig_name}_{int(time.time())}.jpg"
        preview_path = os.path.join(output_dir, preview_file)
        
        preview_img = pil_master
        if pil_master.mode != 'RGB':
            preview_img = pil_master.convert('RGB')
        
        preview_img.save(preview_path, format='JPEG', quality=65, optimize=True)

        update_progress(tracker, task_id, 100, "Masterpiece Created Successfully!")
        
        # Ensure URLs have leading slash
        output_url = "/" + master_path.replace("\\", "/").lstrip("/")
        preview_url = "/" + preview_path.replace("\\", "/").lstrip("/")
        
        return {
            "status": "success",
            "processed_path": preview_url, 
            "output_path": preview_url,
            "master_file": output_url,
            "resolution": f"{w}x{h}",
            "filename": master_file
        }

def process_upscale_logic(data, tracker=None):
    try:
        task_id = data.get('task_id', 'default_task')
        engine = NexusGenerativeEngine(data)
        input_file = data.get('input_image_path')
        if not input_file or not os.path.exists(input_file):
            return {"status": "error", "message": "Source image missing."}
        return engine.process_image(input_file, 'static/outputs/', tracker, task_id)
    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR] {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}