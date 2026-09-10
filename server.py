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

# Exclusive non-European family keywords to filter out false positives in Europe
NON_EUROPEAN_KEYWORDS = [
    'antpitta', 'cupwing', 'hummingbird', 'toucan', 'tanager', 'antbird', 
    'sunbird', 'tody', 'woodcreeper', 'manakin', 'cotinga', 'motmot', 
    'puffbird', 'jacamar', 'hornero', 'spinetail', 'barbet', 'honeyeater'
]

def get_ram_usage_mb():
    """Returns current process RAM usage in megabytes."""
    try:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        return 0.0

def get_country_code(lat, lon):
    """Fast, zero-dependency bounding box lookup for GPS coordinates."""
    if lat is None or lon is None:
        return "Global"
    try:
        lat, lon = float(lat), float(lon)
    except (ValueError, TypeError):
        return "Global"

    # Western Europe (BE, NL, FR, DE, UK, etc.)
    if 35.0 <= lat <= 60.0 and -10.0 <= lon <= 30.0:
        if 49.5 <= lat <= 51.5 and 2.5 <= lon <= 6.5:
            return "BE"
        elif 50.7 <= lat <= 53.6 and 3.3 <= lon <= 7.2:
            return "NL"
        elif 41.3 <= lat <= 51.1 and -5.1 <= lon <= 9.6:
            return "FR"
        elif 47.3 <= lat <= 55.1 and 5.8 <= lon <= 15.0:
            return "DE"
        elif 49.9 <= lat <= 60.9 and -8.6 <= lon <= 1.8:
            return "UK"
        return "EU"

    # South America (BR, AR, CO, PE, CL, etc.)
    if -56.0 <= lat <= 13.0 and -82.0 <= lon <= -34.0:
        if -33.7 <= lat <= 5.3 and -73.9 <= lon <= -34.7:
            return "BR"
        elif -55.1 <= lat <= -21.8 and -73.6 <= lon <= -53.6:
            return "AR"
        elif -4.2 <= lat <= 12.5 and -79.0 <= lon <= -66.8:
            return "CO"
        return "SA"

    # North America
    if 24.5 <= lat <= 49.0 and -125.0 <= lon <= -66.9:
        return "US"

    return "Global"

def is_species_plausible_for_region(common_name, region_code, raw_prob):
    """Filters out obvious non-native families for low/medium confidence detections."""
    if region_code in ["BE", "NL", "FR", "DE", "UK", "EU"]:
        name_lower = common_name.lower()
        # If confidence is below 80% and it's a known endemic South American / Asian family, reject it
        if raw_prob < 0.80 and any(keyword in name_lower for keyword in NON_EUROPEAN_KEYWORDS):
            return False
    return True

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

def sigmoid(x):
    """Converts raw model logits into true probabilities (0.0 to 1.0)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

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
    country_code = get_country_code(lat, lon)
    
    print(f"\n📡 [{req_id}] New Request | GPS: ({lat}, {lon}) -> Region: {country_code} | Start RAM: {current_ram} MB", flush=True)

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
        step_samples = 96000   # 2.0 second stride (3 overlapping windows)

        if len(sig) < min_samples:
            sig = np.pad(sig, (0, min_samples - len(sig)))

        chunks = []
        for i in range(0, len(sig) - min_samples + 1, step_samples):
            chunks.append(sig[i:i + min_samples])
        if not chunks:
            chunks.append(sig[:min_samples])

        results_map = {}
        MIN_PROBABILITY = 0.02 

        for chunk in chunks:
            in_data = np.expand_dims(chunk, axis=0).astype(np.float32)
            interpreter.set_tensor(INPUT_DETAILS[0]['index'], in_data)
            interpreter.invoke()
            
            raw_output = interpreter.get_tensor(OUTPUT_DETAILS[0]['index'])[0]
            probs = sigmoid(raw_output) if np.max(raw_output) > 1.0 or np.min(raw_output) < 0.0 else raw_output
            
            for idx, prob in enumerate(probs):
                p_val = float(prob)
                if p_val >= MIN_PROBABILITY:
                    label = SPECIES_LABELS[idx] if idx < len(SPECIES_LABELS) else f"Species_{idx}"
                    if label not in results_map or p_val > results_map[label]:
                        results_map[label] = p_val

        formatted_results = []
        for label, score in results_map.items():
            parts = label.split('_')
            species_code = parts[0] if len(parts) > 0 else label
            common_name = parts[1] if len(parts) > 1 else label
            scientific_name = parts[2] if len(parts) > 2 else common_name

            raw_prob = float(score)

            # Apply geographical sanity filter
            if not is_species_plausible_for_region(common_name, country_code, raw_prob):
                continue

            boosted_prob = max(raw_prob, 0.25) if raw_prob >= 0.02 else raw_prob
            final_score = round(boosted_prob, 3)

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
                "probability": final_score,
                "country": country_code
            })

        formatted_results = sorted(formatted_results, key=lambda x: x['confidence'], reverse=True)[:5]
        
        if formatted_results:
            species_summary = ", ".join([f"{item['commonName']} ({item['confidence']})" for item in formatted_results])
            print(f"🎯 [{req_id}] Identified [{country_code}] ({len(formatted_results)}): {species_summary}", flush=True)
        else:
            print(f"🎯 [{req_id}] No species met threshold.", flush=True)

        total_time_ms = round((time.perf_counter() - req_start_time) * 1000, 2)
        print(f"📊 [{req_id}] Complete in {total_time_ms}ms ({round(total_time_ms/1000, 2)}s) | Evaluated {len(chunks)} windows | Region: {country_code} | Peak RAM: {get_ram_usage_mb()} MB", flush=True)

        response.headers['Access-Control-Allow-Origin'] = '*' 
        return {
            "results": formatted_results,
            "predictions": formatted_results,
            "birds": formatted_results,
            "detections": formatted_results,
            "success": True,
            "status": "success",
            "country": country_code,
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
