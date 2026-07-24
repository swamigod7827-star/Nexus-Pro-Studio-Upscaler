import os
import uuid
import threading
import traceback
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from logic.core_engine import process_upscale_logic
import logging
from PIL import Image

app = Flask(__name__, static_folder='static')
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

progress_tracker = {}
task_results = {}

@app.route('/api/process-upscale', methods=['POST'])
def process_upscale():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file found.'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file.'})

        task_id = request.form.get('task_id', str(uuid.uuid4()))

        progress_tracker[task_id] = {"percent": 5, "log": "Image Uploaded. Waking up Cloud AI Orchestrator..."}
        input_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{file.filename}")
        file.save(input_path)

        payload = {
            'input_image_path': input_path,
            'task_id': task_id,
            'factor': request.form.get('factor', '4X Standard Upscale'),
            'model': request.form.get('model', 'RealESRGAN v4 (General)'),
            'mode': request.form.get('mode', 'Cloud GPU (Fast)'),
            'settings': request.form.get('settings', '{}'),
            'face': request.form.get('face', '{}'),
            'export': request.form.get('export', '{}')
        }

        def run_task():
            try:
                result = process_upscale_logic(payload, progress_tracker)
                task_results[task_id] = result
            except Exception as e:
                print(f"\n[CRITICAL ERROR in background] {e}\n{traceback.format_exc()}")
                task_results[task_id] = {"status": "error", "message": str(e)}
                progress_tracker[task_id] = {"percent": 100, "log": "Failed."}

        threading.Thread(target=run_task).start()
        
        return jsonify({"status": "processing", "task_id": task_id})

    except Exception as e:
        print(f"\n[SERVER ERROR] {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/progress', methods=['GET'])
def get_progress():
    task_id = request.args.get('task_id')
    if task_id and task_id in progress_tracker:
        return jsonify(progress_tracker[task_id])
    return jsonify({"percent": 0, "log": "Initializing Backend Engine..."})

@app.route('/api/result', methods=['GET'])
def get_result():
    task_id = request.args.get('task_id')
    res = task_results.get(task_id)
    if res:
        return jsonify(res)
    return jsonify({"status": "processing"})

@app.route('/api/merge-pdfs', methods=['POST'])
def merge_pdfs():
    try:
        data = request.json
        file_urls = data.get('files', [])
        if not file_urls:
            return jsonify({"status": "error", "message": "No files provided."})

        merged_pdf_path = os.path.join(OUTPUT_FOLDER, f"Merged_Batch_{uuid.uuid4().hex[:8]}.pdf")
        
        images = []
        for url in file_urls:
            filename = url.split('/')[-1].split('?')[0]
            local_filepath = os.path.join(OUTPUT_FOLDER, filename)
            
            if os.path.exists(local_filepath):
                try:
                    img = Image.open(local_filepath)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Error opening image {local_filepath} for PDF merge: {e}")
            else:
                print(f"Warning: File not found for PDF merge: {local_filepath}")

        if not images:
            return jsonify({"status": "error", "message": "Could not read any valid images to merge."})

        first_image = images[0]
        other_images = images[1:]
        
        first_image.save(
            merged_pdf_path,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=other_images
        )
        
        return jsonify({"status": "success", "merged_url": f"/{merged_pdf_path}"})

    except Exception as e:
        print(f"Merge PDF Error: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/static/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/dynamic-preview')
def dynamic_preview():
    import io
    from PIL import Image
    path = request.args.get('path')
    if not path:
        return "Path missing", 400
    
    local_path = path.lstrip('/')
    if not os.path.exists(local_path):
        return "File not found", 404
        
    try:
        img = Image.open(local_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 NEXUS CLOUD WORKER INITIALIZED")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
