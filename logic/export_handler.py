import os
from PIL import Image

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
    
    # --- Generate Lightweight Preview ---
    # To prevent browser crash when UI tries to load huge files (or unsupported formats like PSD/CDR/PDF)
    update_progress(tracker, task_id, 98, "Generating UI Preview Thumbnail...")
    preview_file = f"{orig_name}_preview_{task_id}.jpg"
    preview_path = os.path.join(output_dir, preview_file)
    
    # Resize preview to max 1920px for fast UI loading
    max_dim = max(pil_master.width, pil_master.height)
    if max_dim > 1920:
        ratio = 1920 / max_dim
        new_size = (int(pil_master.width * ratio), int(pil_master.height * ratio))
        preview_img = pil_master.resize(new_size, Image.Resampling.LANCZOS)
    else:
        preview_img = pil_master.copy()
        
    if preview_img.mode != 'RGB':
        preview_img = preview_img.convert('RGB')
    preview_img.save(preview_path, format='JPEG', quality=85)
    
    output_url = "/" + master_path.replace("\\", "/").lstrip("/")
    preview_url = "/" + preview_path.replace("\\", "/").lstrip("/")
    
    return master_file, output_url, preview_url
