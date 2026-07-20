import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logic_ai_upscaler

app = Flask(__name__, static_folder='static')
CORS(app)

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
@app.route('/api/progress', methods=['GET'])
def get_progress():
    task_id = request.args.get('task_id')
    if task_id and task_id in progress_tracker:
        return jsonify(progress_tracker[task_id])
    return jsonify({"percent": 0, "log": "Initializing Backend Engine..."})

@app.route('/api/process-upscale', methods=['POST'])
def process_upscale():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file found.'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file.'})

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # Catch the Task ID sent by UI
        task_id = request.form.get('task_id', 'default_task')
        
        # Initialize progress at 5%
        progress_tracker[task_id] = {"percent": 5, "log": "Image Uploaded. Waking up AI Orchestrator..."}

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

        # 🚀 CLOUD GPU PROXY ROUTING
        colab_url = request.form.get('colab_url', None)
        if colab_url:
            progress_tracker[task_id] = {"percent": 10, "log": "Connecting to Cloud GPU Proxy..."}
            try:
                with open(input_path, 'rb') as f:
                    file_bytes = f.read()
                    
                files = {'image': (file.filename, file_bytes, file.mimetype)}
                colab_endpoint = colab_url.rstrip('/') + '/api/process-upscale'
                progress_tracker[task_id] = {"percent": 30, "log": "Uploading image to Cloud GPU..."}
                
                form_data = request.form.to_dict()
                if 'colab_url' in form_data:
                    del form_data['colab_url']
                
                # Add Bypass headers for localtunnel/cloudflare
                headers = {
                    'Bypass-Tunnel-Reminder': 'true',
                    'User-Agent': 'curl/7.68.0'
                }
                response = requests.post(colab_endpoint, files=files, data=form_data, headers=headers)
                
                if response.status_code == 200:
                    colab_json = response.json()
                    if colab_json.get('status') == 'success':
                        progress_tracker[task_id] = {"percent": 90, "log": "Downloading upscaled output from Cloud..."}
                        
                        remote_image_path = colab_json.get('output_path') or colab_json.get('processed_path')
                        image_url = colab_url.rstrip('/') + remote_image_path
                        
                        img_response = requests.get(image_url, headers=headers)
                        if img_response.status_code == 200:
                            import time
                            orig_name = os.path.splitext(file.filename)[0]
                            master_file = f"{orig_name}_ColabGPU_{int(time.time())}.jpg"
                            master_path = os.path.join('static/outputs', master_file)
                            
                            with open(master_path, 'wb') as out_f:
                                out_f.write(img_response.content)
                                
                            progress_tracker[task_id] = {"percent": 100, "log": "Process Complete!"}
                            output_url_local = "/" + master_path.replace("\\", "/")
                            return jsonify({
                                "status": "success", 
                                "processed_path": output_url_local,
                                "output_path": output_url_local,
                                "master_file": output_url_local,
                                "filename": master_file
                            })
                        else:
                            raise Exception("Failed to download image from Colab.")
                    else:
                        raise Exception(colab_json.get('message', 'Unknown Cloud Error'))
                else:
                    error_msg = f"API returned {response.status_code}: {response.text[:150]}"
                    raise Exception(error_msg)
            except Exception as e:
                progress_tracker[task_id] = {"percent": 0, "log": f"Cloud Error: {str(e)}"}
                return jsonify({'status': 'error', 'message': f"Cloud GPU Error: {str(e)}"})

        # Send the tracker reference to the LOCAL Orchestrator
        result = logic_ai_upscaler.process_upscale_logic(payload, progress_tracker)
        
        # Clean up tracker memory once task is fully complete
        if task_id in progress_tracker:
            del progress_tracker[task_id]

        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"\n[SERVER ERROR] {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)})



if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 NEXUS PRO SERVER INITIALIZED")
    print("🔗 UI accessible at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)