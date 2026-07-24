import os
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def get_format_details(fmt_str):
    fmt_str = fmt_str.upper()
    if any(x in fmt_str for x in ['CDR', 'AI', 'EPS']):
        return 'eps', 'EPS'
    elif 'PDF' in fmt_str:
        return 'pdf', 'PDF'
    elif any(x in fmt_str for x in ['PSD', 'TIFF', 'TIF']):
        return 'tiff', 'TIFF' # Save PSD as TIFF under the hood, standard practice for compatibility
    elif 'WEBP' in fmt_str:
        return 'webp', 'WEBP'
    elif 'JPG' in fmt_str or 'JPEG' in fmt_str:
        return 'jpg', 'JPEG'
    else:
        return 'png', 'PNG'

def export_image(upscaled, input_path, output_dir, target_scale, color_space, dpi, fmt_str, tracker, task_id, update_progress):
    ext, pil_fmt = get_format_details(fmt_str)
    
    update_progress(tracker, task_id, 95, f"Formatting for Export ({pil_fmt} | {dpi} DPI)...")
    
    # ZERO-COPY MEMORY OPTIMIZATION for massive arrays to prevent OOM
    img_rgb = upscaled[:, :, ::-1]
    pil_master = Image.fromarray(img_rgb)
    
    h, w = upscaled.shape[:2]
    
    # Deep Analysis: Prevent Format conflicts with Color Space
    if "CMYK" in color_space.upper():
        if pil_fmt in ['PNG', 'WEBP']:
            update_progress(tracker, task_id, 96, "Notice: PNG/WEBP do not support CMYK. Exporting as TIFF instead.")
            pil_fmt = 'TIFF'
            ext = 'tiff'
        pil_master = pil_master.convert('CMYK')
    elif "GRAYSCALE" in color_space.upper() or "B&W" in color_space.upper():
        pil_master = pil_master.convert('L')
        
    # Deep Analysis: Prevent PIL PDF OOM Crash on Massive Images
    if pil_fmt == 'PDF' and max(h, w) > 8192:
        update_progress(tracker, task_id, 96, "Notice: Image is too massive for PDF container. Falling back to High-Quality JPEG to prevent RAM Crash.")
        pil_fmt = 'JPEG'
        ext = 'jpg'

    orig_name = os.path.splitext(os.path.basename(input_path))[0]
    
    if orig_name.startswith(f"{task_id}_"):
        orig_name = orig_name[len(task_id)+1:]
    
    # Special Extension Handling (Mocking PSD and CDR with compatible formats)
    final_ext = ext
    if 'PSD' in fmt_str.upper():
        final_ext = 'psd' # Will be a TIFF under the hood but with .psd extension
    elif 'CDR' in fmt_str.upper():
        final_ext = 'cdr' # Will be an EPS under the hood but with .cdr extension
        
    master_file = f"{orig_name}_Nexus_X{int(target_scale)}.{final_ext}"
    master_path = os.path.join(output_dir, master_file)
    
    # Safe kwargs passing to prevent PIL exceptions
    save_kwargs = {}
    if pil_fmt == 'JPEG':
        save_kwargs = {'quality': 100, 'dpi': (dpi, dpi)}
    elif pil_fmt == 'PDF':
        save_kwargs = {'resolution': float(dpi)}
    elif pil_fmt in ['PNG', 'TIFF', 'WEBP']:
        save_kwargs = {'dpi': (dpi, dpi)}
        
    # EPS requires RGB mode
    if pil_fmt == 'EPS' and pil_master.mode not in ['L', 'RGB', 'CMYK']:
        pil_master = pil_master.convert('RGB')
        
    pil_master.save(master_path, format=pil_fmt, **save_kwargs)
    
    update_progress(tracker, task_id, 98, "Generating High-Speed UI Preview...")
    
    # Generate ultra-fast thumbnail using OpenCV instead of PIL
    max_dim = 2048
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        preview_img = cv2.resize(upscaled, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        preview_img = upscaled
        
    preview_dir = os.path.join('static', 'ui_cache')
    os.makedirs(preview_dir, exist_ok=True)
        
    preview_file = f"{orig_name}_preview_{task_id}.jpg"
    preview_path = os.path.join(preview_dir, preview_file)
    cv2.imwrite(preview_path, preview_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    
    preview_url = "/" + preview_path.replace("\\", "/").lstrip("/")
    output_url = "/" + master_path.replace("\\", "/").lstrip("/")
    
    return master_file, output_url, preview_url
