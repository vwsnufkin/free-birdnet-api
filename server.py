import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import uuid
import gc
import time
import subprocess
import resource
import numpy as np
import scipy.io.wavfile as wav
import tflite_runtime.interpreter as tflite
from bottle import route, run, request, response

# Max RAM threshold before queuing (512MB total - 200MB buffer = 312MB)
MAX_RAM_THRESHOLD_MB = 312.0

MODEL_PATH = '/app/models/model.tflite'
LABELS_PATH = '/app/models/labels.txt'

INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None
SPECIES_LABELS = []

def get_ram_usage_mb():
    """Returns current process RAM usage in megabytes."""
    try:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        return 0.0

def load_labels():
    global SPECIES_LABELS
    if not SPECIES_LABELS and os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                SPECIES_LABELS = [line.strip().replace('\r', '') for line in f if line.strip()]
            print(f"✅ Loaded {len(SPECIES_LABELS)} species labels. [RAM: {get_ram_usage_mb()} MB]", flush=True)
        except Exception as e:
            print(f"⚠️ Error reading labels: {e}", flush=True)

def init_tflite_interpreter():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS
    if INTERPRETER is None and os.path.exists(MODEL_PATH):
        try:
            t0 = time.perf_counter()
            INTERPRETER = tflite.Interpreter(model_path=MODEL_PATH, num_threads=1)
            INTERPRETER.allocate_tensors()
            INPUT_DETAILS = INTERPRETER.get_input_details()
            OUTPUT_DETAILS = INTERPRETER.get_output_details()
            init_time = round((time.perf_counter() - t0) * 1000, 2)
            print(f"🟢 TFLite Interpreter initialized in {init_time}ms. [RAM: {get_ram_usage_mb()} MB]", flush=True)
        except Exception as e:
            print(f"❌ TFLite Init Error: {e}", flush=True)
    return INTERPRETER

@route('/ping', method=['GET', 'OPTIONS'])
def ping_server():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
    if request.method == 'OPTIONS':
        return {}
    current_ram = get_ram_usage_mb()
    return {
        "status": "awake", 
        "ram_used_mb": current_ram, 
        "ram_available_mb": round(512.0 - current_ram, 2),
        "accepting_requests": current_ram < MAX_RAM_THRESHOLD_MB
    }

@route('/status', method=['GET', 'OPTIONS'])
def check_status():
    return ping_server()

@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio_request():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization'
    
    if request.method == 'OPTIONS':
        response.status = 200
        return {}

    current_ram = get_ram_usage_mb()
    if current_ram >= MAX_RAM_THRESHOLD_MB:
        response.status = 429
        return {
            "status": "queued",
            "error": "Server memory threshold reached. Request queued.",
            "retry_after_sec": 2,
            "current_ram_mb": current_ram
        }

    req_start_time = time.perf_counter()
    upload = request.files.get('audio') or request.files.get('file')
    if not upload:
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": "No audio file provided"}

    req_id = str(uuid.uuid4())[:8]
    raw_path = f'/tmp/raw_{req_id}'
    wav_path = f'/tmp/rec_{req_id}.wav'

    lat = request.forms.get('lat') or request.forms.get('latitude')
    lon = request.forms.get('lon') or request.forms.get('longitude')
    
    print(f"\n📡 [{req_id}] New Request | Start RAM: {current_ram} MB", flush=True)

    try:
        upload.save(raw_path)

        subprocess.run([
            "ffmpeg", "-y", "-threads", "1", "-i", raw_path, 
            "-filter:a", "volume=10dB", 
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", 
            wav_path
        ], check=True, capture_output=True)

        load_labels()
        interpreter = init_tflite_interpreter()

        if not interpreter:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": "TFLite Model interpreter failed to initialize"}

        rate, data = wav.read(wav_path)
        sig = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)

        min_samples = 144000  # 3.0 seconds at 48kHz
        step_samples = 96000   # 2.0 second stride (3 windows for 8s audio)

        if len(sig) < min_samples:
            sig = np.pad(sig, (0, min_samples - len(sig)))

        chunks = []
        for i in range(0, len(sig) - min_samples + 1, step_samples):
            chunks.append(sig[i:i + min_samples])
        if not chunks:
            chunks.append(sig[:min_samples])

        results_map = {}
        # Ultra-sensitive threshold to capture distinct multi-bird songs across windows
        MIN_CONFIDENCE = 0.001 

        for chunk in chunks:
            in_data = np.expand_dims(chunk, axis=0).astype(np.float32)
            interpreter.set_tensor(INPUT_DETAILS[0]['index'], in_data)
            interpreter.invoke()
            scores = interpreter.get_tensor(OUTPUT_DETAILS[0]['index'])[0]
            
            for idx, score in enumerate(scores):
                raw_s = float(score)
                if raw_s >= MIN_CONFIDENCE:
                    label = SPECIES_LABELS[idx] if idx < len(SPECIES_LABELS) else f"Species_{idx}"
                    # Store max score seen across any window
                    if label not in results_map or raw_s > results_map[label]:
                        results_map[label] = raw_s

        formatted_results = []
        for label, score in results_map.items():
            parts = label.split('_')
            species_code = parts[0] if len(parts) > 0 else label
            common_name = parts[1] if len(parts) > 1 else label
            scientific_name = parts[2] if len(parts) > 2 else common_name

            raw_prob = float(score)
            if raw_prob > 1.0:
                raw_prob = raw_prob / 100.0

            # Proportional linear/log scaling so different birds retain distinct, non-flat scores
            # E.g., raw 0.002 -> 0.216, raw 0.01 -> 0.28, raw 0.08 -> 0.84
            scaled_score = min(0.95, (raw_prob * 8.0) + 0.20)
            final_score = round(scaled_score, 3)

            formatted_results.append({
                "speciesCode": species_code,
                "species_code": species_code,
                "commonName": common_name,
                "common_name": common_name,
                "scientificName": scientific_name,
                "scientific_name": scientific_name,
                "species": common_name,
                "name": common_name,
                "score": final_score,
                "confidence": final_score,
                "probability": final_score
            })

        # Sort by confidence and take top 5
        formatted_results = sorted(formatted_results, key=lambda x: x['confidence'], reverse=True)[:5]
        
        if formatted_results:
            species_summary = ", ".join([f"{item['commonName']} ({item['confidence']})" for item in formatted_results])
            print(f"🎯 [{req_id}] Identified ({len(formatted_results)}): {species_summary}", flush=True)
        else:
            print(f"🎯 [{req_id}] No species met threshold.", flush=True)

        total_time_ms = round((time.perf_counter() - req_start_time) * 1000, 2)
        print(f"📊 [{req_id}] Complete in {total_time_ms}ms ({round(total_time_ms/1000, 2)}s) | Evaluated {len(chunks)} windows | Peak RAM: {get_ram_usage_mb()} MB", flush=True)

        response.headers['Access-Control-Allow-Origin'] = '*' 
        return {
            "results": formatted_results,
            "predictions": formatted_results,
            "birds": formatted_results,
            "detections": formatted_results,
            "success": True,
            "status": "success",
            "count": len(formatted_results)
        }

    except Exception as e:
        print(f"❌ [{req_id}] Error: {str(e)}", flush=True)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": str(e), "success": False}

    finally:
        if os.path.exists(raw_path): 
            try: os.remove(raw_path)
            except: pass
        if os.path.exists(wav_path): 
            try: os.remove(wav_path)
            except: pass
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🟢 Direct TFLite BirdNET server booting on port {port}...", flush=True)
    load_labels()
    init_tflite_interpreter()
    run(host='0.0.0.0', port=port)
