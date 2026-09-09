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

def locate_model_files():
    """Locates ONNX model file and labels.txt file on disk."""
    model_path = '/app/models/model.onnx'
    labels_path = '/app/models/labels.txt'

    if not os.path.exists(model_path):
        candidates = glob.glob('/root/.cache/**/*.onnx', recursive=True) + \
                     glob.glob('/usr/local/lib/python3.11/site-packages/**/*.onnx', recursive=True) + \
                     glob.glob('/app/**/*.onnx', recursive=True)
        for c in candidates:
            if os.path.getsize(c) > 10000000:  # > 10MB
                model_path = c
                break

    if not os.path.exists(labels_path):
        candidates = glob.glob('/root/.cache/**/*label*.txt', recursive=True) + \
                     glob.glob('/usr/local/lib/python3.11/site-packages/**/*label*.txt', recursive=True) + \
                     glob.glob('/app/**/*label*.txt', recursive=True)
        if candidates:
            labels_path = candidates[0]

    return model_path, labels_path

MODEL_PATH, LABELS_PATH = locate_model_files()
ORT_SESSION = None
SPECIES_LABELS = []

def load_labels():
    global SPECIES_LABELS, LABELS_PATH
    if not SPECIES_LABELS:
        if not LABELS_PATH or not os.path.exists(LABELS_PATH):
            _, LABELS_PATH = locate_model_files()
            
        if LABELS_PATH and os.path.exists(LABELS_PATH):
            try:
                with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                    SPECIES_LABELS = [line.strip() for line in f if line.strip()]
                print(f"✅ Loaded {len(SPECIES_LABELS)} species labels from {LABELS_PATH}", flush=True)
            except Exception as e:
                print(f"⚠️ Error reading species labels: {e}", flush=True)

def init_onnx_session():
    global ORT_SESSION, MODEL_PATH
    if ORT_SESSION is None:
        if not MODEL_PATH or not os.path.exists(MODEL_PATH):
            MODEL_PATH, _ = locate_model_files()
            
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            ORT_SESSION = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
            print(f"🟢 Direct ONNX Session initialized from {MODEL_PATH}", flush=True)
        else:
            print(f"❌ Could not resolve valid ONNX model file path on disk!", flush=True)
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

        print(f"🚀 [{req_id[:8]}] Executing direct ONNX inference...", flush=True)

        load_labels()
        session = init_onnx_session()

        if not session:
            print(f"❌ [{req_id[:8]}] ONNX session failed to start.", flush=True)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"results": []}

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

        input_name = session.get_inputs()[0].name
        results_map = {}

        for chunk in chunks:
            in_data = np.expand_dims(chunk, axis=0).astype(np.float32)
            outputs = session.run(None, {input_name: in_data})
            scores = outputs[0][0]
            
            for idx, score in enumerate(scores):
                if score >= 0.03:
                    label = SPECIES_LABELS[idx] if idx < len(SPECIES_LABELS) else f"Species_{idx}"
                    if label not in results_map or score > results_map[label]:
                        results_map[label] = float(score)

        formatted_results = []
        for label, score in results_map.items():
            parts = label.split('_')
            sci_name = parts[0] if len(parts) > 0 else label
            com_name = parts[1] if len(parts) > 1 else label
            
            formatted_results.append({
                "speciesCode": sci_name,
                "commonName": com_name,
                "score": round(score, 3)
            })

        formatted_results = sorted(formatted_results, key=lambda x: x['score'], reverse=True)[:5]
        
        print(f"🎉 [{req_id[:8]}] Success! Identified top {len(formatted_results)} species.", flush=True)
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
    print(f"🟢 Direct ONNX BirdNET server booting on port {port}...", flush=True)
    load_labels()
    init_onnx_session()
    run(host='0.0.0.0', port=port)
