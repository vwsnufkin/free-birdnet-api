import os
os.environ['OMP_NUM_THREADS'] = '1'

import csv
import sys
import glob
import uuid
import gc
import subprocess
import numpy as np
import scipy.io.wavfile as wav
import onnxruntime as ort
from bottle import route, run, request, response

IS_BUSY = False

# Direct path to model downloaded in Dockerfile
LOCAL_MODEL = '/app/models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx'

def locate_onnx_model():
    if os.path.exists(LOCAL_MODEL):
        return LOCAL_MODEL
    possible_paths = glob.glob('/app/**/*.onnx', recursive=True) + \
                     glob.glob('/usr/local/lib/python3.11/site-packages/**/*.onnx', recursive=True)
    if possible_paths:
        return possible_paths[0]
    return None

ORT_SESSION = None

def get_onnx_session():
    global ORT_SESSION
    if ORT_SESSION is None:
        model_path = locate_onnx_model()
        if model_path and os.path.exists(model_path):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            ORT_SESSION = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
            print(f"✅ ONNX model loaded cleanly into memory from {model_path}", flush=True)
    return ORT_SESSION

@route('/ping', method=['GET', 'OPTIONS'])
def ping_server():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
    if request.method == 'OPTIONS':
        return {}
    return {"status": "awake", "busy": IS_BUSY}

@route('/status', method=['GET', 'OPTIONS'])
def check_status():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
    if request.method == 'OPTIONS':
        return {}
    return {"status": "awake", "busy": IS_BUSY}

@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio_request():
    global IS_BUSY
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization'
    
    if request.method == 'OPTIONS':
        response.status = 200
        return {}

    if IS_BUSY:
        response.status = 429
        return {"error": "Server is currently busy analyzing audio. Please try again in a few seconds."}

    print("📡 Incoming audio request received!", flush=True)

    upload = request.files.get('audio')
    if not upload:
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": "No audio file provided"}

    try:
        user_lat = float(request.forms.get('lat', '50.85'))
        user_lon = float(request.forms.get('lon', '4.35'))
    except ValueError:
        user_lat, user_lon = 50.85, 4.35

    print(f"🌍 GPS Location received: Lat {user_lat}, Lon {user_lon}", flush=True)

    req_id = str(uuid.uuid4())
    raw_path = f'/tmp/raw_{req_id}'
    wav_path = f'/tmp/rec_{req_id}.wav'

    try:
        IS_BUSY = True
        upload.save(raw_path)
        print(f"✅ [{req_id[:8]}] Audio file saved. Converting format...", flush=True)

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path, 
                "-filter:a", "volume=10dB", 
                "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", 
                wav_path
            ], check=True, capture_output=True)
            print(f"✅ [{req_id[:8]}] Audio successfully converted.", flush=True)
        except subprocess.CalledProcessError as e:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": f"Audio Conversion Failed: {e.stderr.decode()}"}

        print(f"🚀 [{req_id[:8]}] Running ONNX inference...", flush=True)

        rate, data = wav.read(wav_path)
        
        if data.dtype == np.int16:
            sig = data.astype(np.float32) / 32768.0
        else:
            sig = data.astype(np.float32)

        min_samples = 144000
        if len(sig) < min_samples:
            sig = np.pad(sig, (0, min_samples - len(sig)))

        chunks = []
        step = 144000
        for i in range(0, len(sig) - min_samples + 1, step):
            chunk = sig[i:i + min_samples]
            chunks.append(chunk)

        if not chunks:
            chunks.append(sig[:min_samples])

        session = get_onnx_session()
        results_map = {}

        if session:
            input_name = session.get_inputs()[0].name
            
            for chunk in chunks:
                in_data = np.expand_dims(chunk, axis=0).astype(np.float32)
                outputs = session.run(None, {input_name: in_data})
                scores = outputs[0][0]
                
                for idx, score in enumerate(scores):
                    if score >= 0.03:
                        sp_name = f"Species_{idx}"
                        if sp_name not in results_map or score > results_map[sp_name]:
                            results_map[sp_name] = float(score)

        formatted_results = []
        for sp, score in results_map.items():
            formatted_results.append({
                "speciesCode": sp,
                "commonName": sp,
                "score": round(score, 3)
            })

        formatted_results = sorted(formatted_results, key=lambda x: x['score'], reverse=True)[:5]
        
        print(f"🎉 [{req_id[:8]}] Success! Returning top {len(formatted_results)} matches.", flush=True)
        response.headers['Access-Control-Allow-Origin'] = '*' 
        return {"results": formatted_results}

    except Exception as e:
        print(f"❌ Analysis error: {e}", flush=True)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"results": []}

    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)
        IS_BUSY = False
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🟢 Pure ONNX light server booting on port {port}...", flush=True)
    run(host='0.0.0.0', port=port)
