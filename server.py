import os
# 🚨 CRITICAL: Set TensorFlow thread limits BEFORE importing any ML libraries!
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
from bottle import route, run, request, response

# 1. Quick ping route to wake up Render as soon as the user visits the page
@route('/ping', method=['GET', 'OPTIONS'])
def ping_server():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
    if request.method == 'OPTIONS':
        return {}
    return {"status": "awake", "message": "BirdNET backend is active and ready!"}

@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio():
    # Global CORS wildcards for cross-origin access from Lovable frontend
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization'
    
    if request.method == 'OPTIONS':
        response.status = 200
        return {}

    print("📡 Incoming audio request received!", flush=True)

    upload = request.files.get('audio')
    if not upload:
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": "No audio file provided"}

    # Catch GPS coordinates sent by mobile app (Defaults to -1 if missing)
    user_lat = request.forms.get('lat', '-1')
    user_lon = request.forms.get('lon', '-1')
    
    print(f"🌍 Location data received: Lat {user_lat}, Lon {user_lon}", flush=True)

    # Isolated paths using UUID
    req_id = str(uuid.uuid4())
    raw_path = f'/tmp/raw_{req_id}'
    wav_path = f'/tmp/rec_{req_id}.wav'
    out_dir = f'/tmp/birds_{req_id}'

    try:
        upload.save(raw_path)
        print(f"✅ [{req_id[:8]}] Audio file saved. Converting format...", flush=True)

        # Boost volume by 5dB to clear muffle issues
        try:
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-filter:a", "volume=5dB", "-ar", "48000", wav_path], check=True, capture_output=True)
            print(f"✅ [{req_id[:8]}] Audio successfully converted and volume boosted.", flush=True)
        except subprocess.CalledProcessError as e:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": f"Audio Conversion Failed. Log: {e.stderr.decode()}"}

        # Pass coordinates to AI to filter impossible foreign species
        cmd = [
            "python", "-m", "birdnet_analyzer.analyze",
            "-o", out_dir,
            "--rtype", "csv",
            "--lat", user_lat,
            "--lon", user_lon,
            "--min_conf", "0.01",
            "--n_workers", "1", 
            wav_path 
        ]
        
        print(f"🚀 [{req_id[:8]}] Launching Cornell AI engine...", flush=True)
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            print(f"✅ [{req_id[:8]}] AI Engine finished processing.", flush=True)
        except subprocess.TimeoutExpired:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": "AI Engine timed out."}
        except Exception as e:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": str(e)}

        if "Columns must be same length as key" in process.stderr:
            print(f"✅ [{req_id[:8]}] Cornell empty-result bug caught. Returning 0 birds.", flush=True)
            response.headers['Access-Control-Allow-Origin'] = '*' 
            return {"results": []}

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
                
                # Sort with highest confidence first
                results = sorted(results, key=lambda x: x['score'], reverse=True)
                
                print(f"🎉 [{req_id[:8]}] Success! Returning all {len(results)} detected bird matches.", flush=True)
                response.headers['Access-Control-Allow-Origin'] = '*' 
                return {"results": results}
            
        print(f"❌ [{req_id[:8]}] Result file was never created by the AI!", flush=True)
        response.headers['Access-Control-Allow-Origin'] = '*' 
        return {"error": f"AI Engine Crash: {process.stderr} | {process.stdout}"}

    finally:
        # 🧹 CLEANUP & GARBAGE COLLECTION
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        
        # Force Python memory release back to system
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🟢 OOM-Optimized single-thread server booting up on port {port}...", flush=True)
    
    # Use standard wsgiref server to guarantee sequential, single-threaded execution under 512MB
    run(host='0.0.0.0', port=port)
