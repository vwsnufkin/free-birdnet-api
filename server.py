import os
import subprocess
import csv
import sys
import shutil
import glob
from bottle import route, run, request, response

@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio():
    # 🚨 SYSTEM LEVEL REPAIR: Force explicit wildcard matching for headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization'
    
    if request.method == 'OPTIONS':
        return {}

    print("📡 Incoming audio request received!", flush=True)

    upload = request.files.get('audio')
    if not upload:
        return {"error": "No audio file provided"}

    # Catch the GPS coordinates sent by your website! (Defaults to -1 if missing)
    user_lat = request.forms.get('lat', '-1')
    user_lon = request.forms.get('lon', '-1')
    
    print(f"🌍 Location data received: Lat {user_lat}, Lon {user_lon}", flush=True)

    raw_path = '/tmp/raw_upload'
    wav_path = '/tmp/recording.wav'
    out_dir = '/tmp/bird_results'
    
    if os.path.exists(raw_path): os.remove(raw_path)
    if os.path.exists(wav_path): os.remove(wav_path)
    if os.path.exists(out_dir): shutil.rmtree(out_dir) 
        
    upload.save(raw_path)
    print("✅ Audio file saved to server. Converting format...", flush=True)

    try:
        # We added a tiny volume boost (-filter:a volume=5dB) to help overcome speaker muffling
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-filter:a", "volume=5dB", "-ar", "48000", wav_path], check=True, capture_output=True)
        print("✅ Audio successfully converted and volume boosted.", flush=True)
    except subprocess.CalledProcessError as e:
        return {"error": f"Audio Conversion Failed. Log: {e.stderr.decode()}"}

    # Pass the real coordinates to the AI so it filters out impossible foreign birds!
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
    
    print(f"🚀 Launching Cornell AI engine...", flush=True)
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        print("✅ AI Engine finished processing.", flush=True)
    except subprocess.TimeoutExpired:
        return {"error": "AI Engine timed out. It needs more than 3 minutes to boot up on this free server."}
    except Exception as e:
        return {"error": str(e)}

    if "Columns must be same length as key" in process.stderr:
        print("✅ Cornell empty-result bug caught. Returning 0 birds.", flush=True)
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
            
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            print(f"🎉 Success! Found {len(results)} birds.", flush=True)
            return {"results": results}
        
    print("❌ Result file was never created by the AI!", flush=True)
    return {"error": f"AI Engine Crash: {process.stderr} | {process.stdout}"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🟢 Server booting up on port {port}...", flush=True)
    run(host='0.0.0.0', port=port)
