import os
import cv2
import json
import torch

from logic.model_manager import get_upsampler
from logic.face_recovery import get_face_enhancer, apply_face_recovery
from logic.adjustments import apply_pre_upscale_adjustments, apply_post_upscale_blend
from logic.upscale_handler import perform_upscale
from logic.export_handler import export_image

def update_progress(tracker, task_id, percent, log_msg):
    if tracker is not None and task_id:
        tracker[task_id] = {"percent": percent, "log": log_msg}
    print(f"[PROGRESS] {percent}%: {log_msg}", flush=True)

class NexusGenerativeEngine:
    def __init__(self, payload):
        print("\n" + "★"*75)
        print("⚙️ NEXUS PRO: GENERATIVE ENGINE (MODULAR) ⚙️")
        print("★"*75)
        
        self.model_name = payload.get('model', 'RealESRGAN v4')
        self.proc_mode = payload.get('mode', 'CPU Precision (Slow)')
        print(f"[*] [UI ROUTER] Selected AI Model: {self.model_name}")
        print(f"[*] [UI ROUTER] Hardware Mode: {self.proc_mode}")
        
        self.device = 'cuda' if ('GPU' in self.proc_mode or 'TensorRT' in self.proc_mode) and torch.cuda.is_available() else 'cpu'
        self.half_precision = True if self.device == 'cuda' else False
        print(f"[*] [AI ENGINE] Hardware Device: {self.device} (FP16: {self.half_precision})")
        
        self.target_scale = int(''.join(filter(str.isdigit, str(payload.get('factor', '4')).split(' ')[0])) or 4)
        
        adv = json.loads(payload.get('settings', '{}'))
        self.strength = int(adv.get('strength', 100)) / 100.0
        self.noise = int(adv.get('noise', 0))
        self.texture = int(adv.get('texture', 50))
        self.artifact_rem = int(adv.get('artifact', 25))
        
        face = json.loads(payload.get('face', '{}'))
        self.face_restore = str(face.get('enabled', 'false')).lower() == 'true'
        
        if any(keyword in self.model_name for keyword in ["Face", "Portrait", "CodeFormer", "GFPGAN"]):
            self.face_restore = True
            print("[*] [UI ROUTER] Auto-enabled Face Recovery for selected portrait model.")
            
        self.face_weight = int(face.get('identity', 85)) / 100.0 
        self.face_skin = int(face.get('skin_tone', 50))
        
        export = json.loads(payload.get('export', '{}'))
        self.dpi = int(''.join(filter(str.isdigit, str(export.get('dpi', '600')).split(' ')[0])) or 600)
        self.color_space = export.get('color_space', 'sRGB')
        self.fmt_str = export.get('format', 'PNG').upper()

    def process_image(self, input_path, output_dir, tracker=None, task_id=None):
        update_progress(tracker, task_id, 10, f"Preparing Input File -> {os.path.basename(input_path)}...")
        img = cv2.imread(input_path)
        if img is None: 
            raise ValueError("Image corrupted or missing.")

        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

        # 1. Advanced Adjustments (Pre)
        img = apply_pre_upscale_adjustments(img, self.artifact_rem, self.texture, self.face_skin)

        update_progress(tracker, task_id, 25, f"Allocating {self.device.upper()} Threads and Loading Architecture...")
        
        # 2. Get Engines
        upsampler = get_upsampler(self.model_name, self.device, self.half_precision)
        face_enhancer = None
        if self.face_restore:
            face_enhancer = get_face_enhancer(self.model_name, self.target_scale, self.device, upsampler)

        # 3. Upscale & Face Recovery
        out_h, out_w = img.shape[0] * self.target_scale, img.shape[1] * self.target_scale
        
        if self.face_restore and face_enhancer is not None:
            if max(out_h, out_w) > 16384:
                update_progress(tracker, task_id, 30, "Applying Generative Face Restoration (Pre-Scaling)...")
                img = apply_face_recovery(img, face_enhancer, self.face_weight)
                
                update_progress(tracker, task_id, 70, f"Processing {self.target_scale}X Upscaling & Generative Enhancements...")
                upscaled = perform_upscale(img, upsampler, self.target_scale)
            else:
                update_progress(tracker, task_id, 30, f"Processing {self.target_scale}X Upscaling & Generative Enhancements...")
                upscaled = perform_upscale(img, upsampler, self.target_scale)
                
                update_progress(tracker, task_id, 70, "Applying Generative Face Restoration...")
                upscaled = apply_face_recovery(upscaled, face_enhancer, self.face_weight)
        else:
            update_progress(tracker, task_id, 30, f"Processing {self.target_scale}X Upscaling & Generative Enhancements...")
            upscaled = perform_upscale(img, upsampler, self.target_scale)

        # 4. Advanced Adjustments (Post)
        if self.strength < 1.0:
            update_progress(tracker, task_id, 80, f"Applying Upscale Strength ({int(self.strength*100)}%)...")
            upscaled = apply_post_upscale_blend(upscaled, img, self.strength)

        # 5. Export Handing (includes Preview Generation)
        master_file, master_url, preview_url = export_image(
            upscaled, input_path, output_dir, self.target_scale, 
            self.color_space, self.dpi, self.fmt_str, tracker, task_id, update_progress
        )
        
        update_progress(tracker, task_id, 99, "Finalizing Output File...")
        update_progress(tracker, task_id, 100, "Masterpiece Created Successfully!")
        
        h, w = upscaled.shape[:2]
        
        return {
            "status": "success",
            "master_file": master_url,
            "preview_file": preview_url,
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
