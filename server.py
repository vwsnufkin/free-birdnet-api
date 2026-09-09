import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import glob
import uuid
import gc
import json
import shutil
import subprocess
import numpy as np
import scipy.io.wavfile as wav
import onnxruntime as ort
from bottle import route, run, request, response

IS_BUSY = False

def run_system_diagnostics():
    """Scans system and package directory to identify missing files and provide fixes."""
    diag = {
        "status": "UNKNOWN",
        "python_version": sys.version,
        "onnx_model": {"found": False, "path": None, "size_bytes": 0},
        "labels_file": {"found": False, "path": None, "count": 0},
        "searched_onnx_paths": [],
        "searched_label_paths": [],
        "disk_space": {},
        "root_cache_contents": [],
        "site_packages_birdnet": [],
        "possible_fixes": []
    }

    # Disk Space Check
    total, used, free = shutil.disk_usage("/")
    diag["disk_space"] = {
        "free_mb": round(free / (1024 * 1024), 2),
        "total_mb": round(total / (1024 * 1024), 2)
    }

    # Search for ONNX models across container
    onnx_patterns = [
        '/app/models/*.onnx',
        '/app/**/*.onnx',
        '/root/.cache/**/*.onnx',
        '/usr/local/lib/python3.11/site-packages/**/*.onnx',
        '/tmp/**/*.onnx'
    ]
    
    found_onnx = []
    for pattern in onnx_patterns:
        matches = glob.glob(pattern, recursive=True)
        diag["searched_onnx_paths"].append({pattern: matches})
        for m in matches:
            found_onnx.append((m, os.path.getsize(m)))

    if found_onnx:
        # Pick largest file (actual model is ~224MB)
        found_onnx.sort(key=lambda x: x[1], reverse=True)
        diag["onnx_model"] = {
            "found": True,
            "path": found_onnx[0][0],
            "size_bytes": found_onnx[0][1],
            "size_mb": round(found_onnx[0][1] / (1024 * 1024), 2)
        }

    # Search for Labels txt
    label_patterns = [
        '/app/models/*label*.txt',
        '/app/**/*label*.txt',
        '/root/.cache/**/*label*.txt',
        '/usr/local/lib/python3.11/site-packages/**/*label*.txt'
    ]
    
    found_labels = []
    for pattern in label_patterns:
        matches = glob.glob(pattern, recursive=True)
        diag["searched_label_paths"].append({pattern: matches})
        for m in matches:
            found_labels.append(m)

    if found_labels:
        try:
            with open(found_labels[0], 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            diag["labels_file"] = {
                "found": True,
                "path": found_labels[0],
                "count": len(lines)
            }
        except Exception as e:
            diag["labels_file"] = {"found": True, "error": str(e)}

    # List cache contents
    if os.path.exists('/root/.cache'):
        diag["root_cache_contents"] = glob.glob('/root/.cache/**/*', recursive=True)[:20]

    # Evaluate Overall Status & Formulate Fixes
    if diag["onnx_model"]["found"] and diag["onnx_model"]["size_bytes"] > 50000000:
        diag["status"] = "HEALTHY"
        diag["possible_fixes"].append("Model file is valid and present. Inference ready.")
    else:
        diag["status"] = "CRITICAL_MISSING_MODEL"
        diag["possible_fixes"].append(
            "NO_MODEL_FILE_ON_DISK: birdnet_analyzer PyPI wheel does not ship with .onnx binary weights. "
            "Fix: Download the binary directly or copy it into /app/models during docker build."
        )

    return diag

# Run diagnostics at boot
BOOT_DIAG = run_system_diagnostics()
print(f"📊 BOOT DIAGNOSTICS: {json.dumps(BOOT_DIAG, indent=2)}", flush=True)

MODEL_PATH = BOOT_DIAG["onnx_model"].get("path")
LABELS_PATH = BOOT_DIAG["labels_file"].get("path")
ORT_SESSION = None
SPECIES_LABELS = []

def load_labels():
    global SPECIES_LABELS, LABELS_PATH
    if not SPECIES_LABELS and LABELS_PATH and os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                SPECIES_LABELS = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded {len(SPECIES_LABELS)} species labels.", flush=True)
        except Exception as e:
            print(f"⚠️ Error reading labels: {e}", flush=True)

def init_onnx_session():
    global ORT_SESSION, MODEL_PATH
    if ORT_SESSION is None and MODEL_PATH and os.path.exists(MODEL_PATH):
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            ORT_SESSION = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
            print(f"🟢 Direct ONNX Session initialized from {MODEL_PATH}", flush=True)
        except Exception as e:
            print(f"❌ ONNX Session Init Error: {e}", flush=True)
    return ORT_SESSION

@route('/debug', method=['GET', 'OPTIONS'])
def debug_endpoint():
    """Returns real-time container filesystem scan and fix recommendations."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
    if request.method == 'OPTIONS':
        return {}
    return run_system_diagnostics()

@route('/ping', method=['GET', 'OPTIONS'])
def ping_server():
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
            # Generate real-time diagnosis if session is missing
            diag = run_system_diagnostics()
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {
                "error": "ONNX Model Session Failed to Start",
                "diagnostics": diag
            }

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
        diag = run_system_diagnostics()
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": str(e), "diagnostics": diag}

    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)
        IS_BUSY = False
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    load_labels()
    init_onnx_session()
    run(host='0.0.0.0', port=port)
