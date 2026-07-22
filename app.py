import os
import requests
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import logic_ai_upscaler

app = Flask(__name__, static_folder='static')
CORS(app)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 🚀 NAYA: Global Dictionary to track live progress of tasks
progress_tracker = {}

@app.route('/')
def index():
    return send_from_directory('.', 'ai_upscaler.html')

# 🚀 NAYA: Real-Time API Endpoint for Progress Polling
import threading
import uuid
import urllib.parse

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
        colab_url = request.form.get('colab_url', None)

        if colab_url:
            # PURE CLOUD MODE: Proxy task to Colab async endpoint
            progress_tracker[task_id] = {"percent": 5, "log": "Connecting to Cloud GPU Proxy...", "colab_url": colab_url}
            try:
                # Read directly from memory
                file_bytes = file.read()
                files = {'image': (file.filename, file_bytes, file.content_type)}
                
                colab_endpoint = colab_url.rstrip('/') + '/api/process-upscale'
                
                form_data = request.form.to_dict()
                if 'colab_url' in form_data:
                    del form_data['colab_url']
                
                headers = {'Bypass-Tunnel-Reminder': 'true', 'User-Agent': 'curl/7.68.0'}
                # Colab runs the same app.py, so it will return {"status": "processing"} instantly
                response = requests.post(colab_endpoint, files=files, data=form_data, headers=headers)
                
                if response.status_code == 200:
                    colab_json = response.json()
                    if colab_json.get('status') == 'processing':
                        return jsonify({"status": "processing", "task_id": task_id})
                    elif colab_json.get('status') == 'success': # fallback if colab runs old app.py
                        remote_image_path = colab_json.get('output_path') or colab_json.get('processed_path')
                        image_url = colab_url.rstrip('/') + remote_image_path
                        proxy_url = f"/proxy-cloud-image?url={urllib.parse.quote(image_url)}"
                        return jsonify({
                            "status": "success", 
                            "processed_path": proxy_url,
                            "output_path": proxy_url,
                            "master_file": proxy_url,
                            "filename": colab_json.get('filename', f'Cloud_{file.filename}')
                        })
                    else:
                        raise Exception(colab_json.get('message', 'Unknown Cloud Error'))
                else:
                    error_msg = f"API returned {response.status_code}: {response.text[:150]}"
                    raise Exception(error_msg)
            except Exception as e:
                progress_tracker[task_id] = {"percent": 0, "log": f"Cloud Error: {str(e)}"}
                return jsonify({'status': 'error', 'message': f"Cloud GPU Error: {str(e)}"})
        
        # LOCAL MODE (or running INSIDE Colab)
        progress_tracker[task_id] = {"percent": 5, "log": "Image Uploaded. Waking up Local AI Orchestrator..."}
        input_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{file.filename}")
        file.save(input_path)

        payload = {
            'input_image_path': input_path,
            'task_id': task_id,
            'factor': request.form.get('factor', '4X Standard Upscale'),
            'model': request.form.get('model', 'RealESRGAN v4 (General)'),
            'mode': request.form.get('mode', 'CPU Precision (Slow)'),
            'settings': request.form.get('settings', '{}'),
            'face': request.form.get('face', '{}'),
            'export': request.form.get('export', '{}')
        }

        # Run in background to prevent browser/tunnel timeout on long tasks
        def run_task():
            try:
                import logic_ai_upscaler
                result = logic_ai_upscaler.process_upscale_logic(payload, progress_tracker)
                task_results[task_id] = result
            except Exception as e:
                import traceback
                print(f"\n[CRITICAL ERROR in background] {e}\n{traceback.format_exc()}")
                task_results[task_id] = {"status": "error", "message": str(e)}
                progress_tracker[task_id] = {"percent": 100, "log": "Failed."}

        threading.Thread(target=run_task).start()
        
        return jsonify({"status": "processing", "task_id": task_id})

    except Exception as e:
        import traceback
        print(f"\n[SERVER ERROR] {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/progress', methods=['GET'])
def get_progress():
    task_id = request.args.get('task_id')
    if task_id and task_id in progress_tracker:
        data = progress_tracker[task_id]
        colab_url = data.get('colab_url')
        
        # If proxying, ask Colab for progress
        if colab_url:
            try:
                headers = {'Bypass-Tunnel-Reminder': 'true', 'User-Agent': 'curl/7.68.0'}
                res = requests.get(f"{colab_url.rstrip('/')}/api/progress?task_id={task_id}", headers=headers, timeout=5)
                return jsonify(res.json())
            except Exception as e:
                return jsonify({"percent": data.get('percent', 10), "log": "Waiting for Cloud GPU response..."})
                
        return jsonify(data)
    return jsonify({"percent": 0, "log": "Initializing Backend Engine..."})

@app.route('/proxy-cloud-image')
def proxy_cloud_image():
    url = request.args.get('url')
    if not url:
        return "URL is missing", 400
    try:
        headers = {'Bypass-Tunnel-Reminder': 'true', 'User-Agent': 'curl/7.68.0'}
        req = requests.get(url, headers=headers, stream=True, timeout=30)
        return Response(stream_with_context(req.iter_content(chunk_size=8192)), content_type=req.headers.get('Content-Type', 'image/jpeg'))
    except Exception as e:
        return str(e), 500

@app.route('/api/result', methods=['GET'])
def get_result():
    task_id = request.args.get('task_id')
    data = progress_tracker.get(task_id, {})
    colab_url = data.get('colab_url')
    
    if colab_url:
        try:
            headers = {'Bypass-Tunnel-Reminder': 'true', 'User-Agent': 'curl/7.68.0'}
            res = requests.get(f"{colab_url.rstrip('/')}/api/result?task_id={task_id}", headers=headers, timeout=10)
            colab_json = res.json()
            if colab_json.get('status') == 'success':
                # Map remote URLs to proxy URLs
                remote_image_path = colab_json.get('output_path') or colab_json.get('processed_path')
                image_url = colab_url.rstrip('/') + remote_image_path
                proxy_url = f"/proxy-cloud-image?url={urllib.parse.quote(image_url)}"
                
                remote_master = colab_json.get('master_file')
                master_url = colab_url.rstrip('/') + remote_master
                proxy_master = f"/proxy-cloud-image?url={urllib.parse.quote(master_url)}"
                
                return jsonify({
                    "status": "success", 
                    "processed_path": proxy_url,
                    "output_path": proxy_url,
                    "master_file": proxy_master,
                    "filename": colab_json.get('filename')
                })
            return jsonify(colab_json)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Cloud Result Fetch Error: {str(e)}"})

    res = task_results.get(task_id)
    if res:
        return jsonify(res)
    return jsonify({"status": "processing"})

@app.route('/api/merge-pdfs', methods=['POST'])
def merge_pdfs():
    try:
        data = request.json
        colab_url = data.get('colab_url')
        file_urls = data.get('files', [])
        
        if not file_urls:
            return jsonify({"status": "error", "message": "No files provided."})

        # Proxy to Cloud GPU if active
        if colab_url:
            headers = {'Bypass-Tunnel-Reminder': 'true', 'User-Agent': 'curl/7.68.0'}
            # Send just the raw paths to cloud
            cloud_payload = {"files": file_urls}
            res = requests.post(f"{colab_url.rstrip('/')}/api/merge-pdfs", json=cloud_payload, headers=headers)
            cloud_json = res.json()
            if cloud_json.get('status') == 'success':
                merged_url = colab_url.rstrip('/') + cloud_json.get('merged_url')
                proxy_url = f"/proxy-cloud-image?url={urllib.parse.quote(merged_url)}"
                return jsonify({"status": "success", "merged_url": proxy_url})
            return jsonify(cloud_json)

        # Local Processing
        from PIL import Image
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
        import traceback
        print(f"Merge PDF Error: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)})



if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 NEXUS PRO SERVER INITIALIZED")
    print("🔗 UI accessible at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)