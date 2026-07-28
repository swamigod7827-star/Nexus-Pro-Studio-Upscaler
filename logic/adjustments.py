import cv2
import numpy as np

def apply_pre_upscale_adjustments(img, artifact_rem, texture, face_skin):
    # Artifact Removal (Denoising)
    if artifact_rem > 0:
        rem_val = max(1, artifact_rem // 3)
        img = cv2.bilateralFilter(img, 5, rem_val, rem_val)
        
    # Texture Control (Sharpening / Softening)
    if texture != 50:
        amount = (texture - 50) / 50.0
        if amount > 0:
            blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
            img = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        else:
            img = cv2.GaussianBlur(img, (0, 0), 1.0 + (-amount * 2.0))
            
    # Skin Tone / Color Temp Correction
    if face_skin != 50:
        shift = (face_skin - 50) / 50.0
        b, g, r = cv2.split(img)
        if shift > 0:
            r = cv2.add(r, int(shift * 15))
            b = cv2.subtract(b, int(shift * 15))
        else:
            b = cv2.add(b, int(-shift * 15))
            r = cv2.subtract(r, int(-shift * 15))
        img = cv2.merge((b, g, r))

    return img

def apply_post_upscale_blend(upscaled, original_img, strength):
    h, w = upscaled.shape[:2]
    
    # 1. Strength Blending (Only applies if original is scaled up and blended)
    if strength < 1.0 and max(h, w) < 16384:
        try:
            base_img = cv2.resize(original_img, (w, h), interpolation=cv2.INTER_CUBIC)
            upscaled = cv2.addWeighted(upscaled, strength, base_img, 1.0 - strength, 0)
        except Exception:
            pass # Skip if RAM spikes
            
    # 2. Smart Grain Injection (Defeats the "Plastic/Painting" AI look)
    # AI models destroy sensor noise. Adding 2-3% noise back makes it look like a real photograph.
    try:
        # Convert to LAB to only add noise to Luminance (avoids color noise artifacts)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Generate Gaussian noise matching the L channel dimensions
        noise = np.zeros(l.shape, dtype=np.int16)
        cv2.randn(noise, 0, 3) # mean 0, stddev 3 (subtle film grain)
        
        # Add noise and clip
        l_noisy = cv2.add(l.astype(np.int16), noise)
        l_noisy = np.clip(l_noisy, 0, 255).astype(np.uint8)
        
        # Merge back
        lab_noisy = cv2.merge((l_noisy, a, b))
        upscaled = cv2.cvtColor(lab_noisy, cv2.COLOR_LAB2BGR)
    except Exception:
        pass # Failsafe
        
    return upscaled
