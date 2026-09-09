import os
# Enforce strict CPU/memory isolation
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import subprocess
import csv
import sys
import shutil
import glob
import uuid
import gc
import runpy
from bottle import route, run, request, response

# Global lock state to signal server status to Lovable
IS_BUSY = False

def warmup_models():
    """Cache species taxonomy and acoustic models at server boot."""
    print("⏳ Warming up BirdNET models on startup...", flush=True)
    try:
        from birdnet_analyzer import species
        # Warmup species cache using local EU coordinates (Brussels)
        species.get_species_list(50.85, 4.35, 0.15)
        print("✅ Species taxonomy warmed up successfully!", flush=True)
    except Exception as e:
        print(f"⚠️ Warmup info: {e}", flush=True)

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
    return {"busy": IS_BUSY}

@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio_request():
    global IS_BUSY
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization'
    
    if request.method == 'OPTIONS':
        response.status = 200
        return {}

    # Lock out concurrent requests to protect RAM
    if IS_BUSY:
        response.status = 429
        return {"error": "Server is currently busy analyzing audio. Please try again in a few seconds."}

    print("📡 Incoming audio request received!", flush=True)

    upload = request.files.get('audio')
    if not upload:
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": "No audio file provided"}

    user_lat = request.forms.get('lat', '-1')
    user_lon = request.forms.get('lon', '-1')
    print(f"🌍 GPS Location received: Lat {user_lat}, Lon {user_lon}", flush=True)

    req_id = str(uuid.uuid4())
    raw_path = f'/tmp/raw_{req_id}'
    wav_path = f'/tmp/rec_{req_id}.wav'
    out_dir = f'/tmp/birds_{req_id}'

    try:
        IS_BUSY = True
        upload.save(raw_path)
        print(f"✅ [{req_id[:8]}] Audio file saved. Converting format...", flush=True)

        try:
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-filter:a", "volume=5dB", "-ar", "48000", wav_path], check=True, capture_output=True)
            print(f"✅ [{req_id[:8]}] Audio successfully converted.", flush=True)
        except subprocess.CalledProcessError as e:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": f"Audio Conversion Failed: {e.stderr.decode()}"}

        print(f"🚀 [{req_id[:8]}] Launching Cornell AI engine with GPS filtering...", flush=True)
        
        old_argv = sys.argv
        
        sys.argv = [
            "birdnet_analyzer.analyze",
            "-o", out_dir,
            "--rtype", "csv",
            "--lat", str(user_lat),
            "--lon", str(user_lon),
            "--min_conf", "0.15",
            "-t", "1",
            wav_path
        ]

        try:
            runpy.run_module('birdnet_analyzer.analyze', run_name='__main__')
            print(f"✅ [{req_id[:8]}] AI Engine finished processing cleanly.", flush=True)
        except SystemExit:
            pass
        except Exception as e:
            print(f"❌ Execution error: {e}", flush=True)
        finally:
            sys.argv = old_argv

        results = []
        if os.path.exists(out_dir):
            csv_files = glob.glob(f"{out_dir}/*.csv")
            if csv_files:
                with open(csv_files[0], 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        results.append({
                            "speciesCode": row.get('Scientific name', ''),
                            "commonName": row.get('Common name', ''),
                            "score": float(row.get('Confidence', 0))
                        })
                
                results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
                print(f"🎉 [{req_id[:8]}] Success! Returning top {len(results)} local matches.", flush=True)
                response.headers['Access-Control-Allow-Origin'] = '*' 
                return {"results": results}
            
        print(f"⚠️ [{req_id[:8]}] No CSV results generated (no bird songs detected above 15% confidence).", flush=True)
        response.headers['Access-Control-Allow-Origin'] = '*' 
        return {"results": []}

    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        IS_BUSY = False
        gc.collect()

if __name__ == '__main__':
    warmup_models()
    port = int(os.environ.get('PORT', 10000))
    print(f"🟢 Location-aware pre-loaded server booting on port {port}...", flush=True)
    run(host='0.0.0.0', port=port)
