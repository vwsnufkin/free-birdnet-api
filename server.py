import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import uuid
import gc
import subprocess
import numpy as np
import scipy.io.wavfile as wav
import onnxruntime as ort
from bottle import route, run, request, response

IS_BUSY = False

MODEL_PATH = '/app/models/model.onnx'
LABELS_PATH = '/app/models/labels.txt'

ORT_SESSION = None
SPECIES_LABELS = []

def load_labels():
    global SPECIES_LABELS
    if not SPECIES_LABELS and os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                SPECIES_LABELS = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded {len(SPECIES_LABELS)} species labels.", flush=True)
        except Exception as e:
            print(f"⚠️ Error reading labels: {e}", flush=True)

def init_onnx_session():
    global ORT_SESSION
    if ORT_SESSION is None and os.path.exists(MODEL_PATH):
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            ORT_SESSION = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
            print(f"🟢 Direct ONNX Session initialized cleanly from {MODEL_PATH}", flush=True)
        except Exception as e:
            print(f"❌ ONNX Session Init Error: {e}", flush=True)
    return ORT_SESSION

@route('/ping', method=['GET', 'OPTIONS'])
def ping_server():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
    if request.method == 'OPTIONS':
        return {}
    return {"status": "awake", "busy": IS_BUSY, "model_ready": ORT_SESSION is not None}

@route('/status', method=['GET', 'OPTIONS'])
def check_status():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
    if request.method == 'OPTIONS':
        return {}
    return {"status": "awake", "busy": IS_BUSY, "model_ready": ORT_SESSION is not None}

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

    upload = request.files.get('audio')
    if not upload:
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": "No audio file provided"}

    req_id = str(uuid.uuid4())
    raw_path = f'/tmp/raw_{req_id}'
    wav_path = f'/tmp/rec_{req_id}.wav'

    try:
        IS_BUSY = True
        upload.save(raw_path)

        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path, 
            "-filter:a", "volume=10dB", 
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", 
            wav_path
        ], check=True, capture_output=True)

        load_labels()
        session = init_onnx_session()

        if not session:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": "ONNX Model session failed to initialize"}

        rate, data = wav.read(wav_path)
        sig = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)

        min_samples = 144000
        if len(sig) < min_samples:
            sig = np.pad(sig, (0, min_samples - len(sig)))

        chunks = [sig[i:i + min_samples] for i in range(0, len(sig) - min_samples + 1, min_samples)]
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
            formatted_results.append({
                "speciesCode": parts[0] if len(parts) > 0 else label,
                "commonName": parts[1] if len(parts) > 1 else label,
                "score": round(score, 3)
            })

        formatted_results = sorted(formatted_results, key=lambda x: x['score'], reverse=True)[:5]
        response.headers['Access-Control-Allow-Origin'] = '*' 
        return {"results": formatted_results}

    except Exception as e:
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": str(e)}

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
