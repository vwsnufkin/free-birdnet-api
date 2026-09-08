import os
# Force strict single-thread TensorFlow settings BEFORE loading libraries
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

# Import BirdNET analyzer CLI runner
try:
    import birdnet_analyzer.analyze as birdnet_cli
except ImportError:
    birdnet_cli = None

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

    user_lat = request.forms.get('lat', '-1')
    user_lon = request.forms.get('lon', '-1')

    req_id = str(uuid.uuid4())
    raw_path = f'/tmp/raw_{req_id}'
    wav_path = f'/tmp/rec_{req_id}.wav'
    out_dir = f'/tmp/birds_{req_id}'

    try:
        upload.save(raw_path)
        print(f"✅ [{req_id[:8]}] Audio file saved. Converting format...", flush=True)

        try:
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-filter:a", "volume=5dB", "-ar", "48000", wav_path], check=True, capture_output=True)
            print(f"✅ [{req_id[:8]}] Audio successfully converted.", flush=True)
        except subprocess.CalledProcessError as e:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": f"Audio Conversion Failed: {e.stderr.decode()}"}

        print(f"🚀 [{req_id[:8]}] Launching Cornell AI engine in-memory (Min Conf: 15%)...", flush=True)
        
        # Save old sys.argv to restore later
        old_argv = sys.argv
        
        # Construct synthetic CLI flags directly inside the existing Python process
        sys.argv = [
            "birdnet_analyzer",
            "-i", wav_path,
            "-o", out_dir,
            "--rtype", "csv",
            "--lat", str(user_lat),
            "--lon", str(user_lon),
            "--min_conf", "0.15",
            "--n_workers", "1"
        ]

        try:
            # 💡 FIX: Check if birdnet_cli is a function or module and execute accordingly
            if callable(birdnet_cli):
                birdnet_cli()
            elif hasattr(birdnet_cli, 'main'):
                birdnet_cli.main()
            else:
                # Direct module execution fallback
                from birdnet_analyzer import client
                client.main()
                
            print(f"✅ [{req_id[:8]}] AI Engine finished processing cleanly.", flush=True)
        except SystemExit:
            # Catch standard sys.exit() calls from CLI parsers
            pass
        except Exception as e:
            print(f"❌ In-process execution error: {e}", flush=True)
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
                
                # Sort by confidence and return top 5
                results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
                
                print(f"🎉 [{req_id[:8]}] Success! Returning top {len(results)} matches.", flush=True)
                response.headers['Access-Control-Allow-Origin'] = '*' 
                return {"results": results}
            
        print(f"⚠️ [{req_id[:8]}] No CSV results generated.", flush=True)
        response.headers['Access-Control-Allow-Origin'] = '*' 
        return {"results": []}

    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🟢 In-process single-thread server booting on port {port}...", flush=True)
    run(host='0.0.0.0', port=port)
