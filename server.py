import os
import subprocess
import csv
import sys
from bottle import route, run, request, response

@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    
    if request.method == 'OPTIONS':
        return {}

    print("📡 Incoming audio request received!", flush=True)

    upload = request.files.get('audio')
    if not upload:
        print("❌ Error: No audio file found in request.", flush=True)
        return {"error": "No audio file provided"}

    raw_path = '/tmp/raw_upload'
    wav_path = '/tmp/recording.wav'
    out_path = '/tmp/result.csv'
    
    for p in [raw_path, wav_path, out_path]:
        if os.path.exists(p): os.remove(p)
        
    upload.save(raw_path)
    print("✅ Audio file saved to server. Converting format...", flush=True)

    try:
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "48000", wav_path], check=True, capture_output=True)
        print("✅ Audio successfully converted to perfect WAV.", flush=True)
    except subprocess.CalledProcessError as e:
        print("❌ ffmpeg conversion failed!", flush=True)
        return {"error": f"Audio Conversion Failed. Log: {e.stderr.decode()}"}

    # 🚨 FIX: Changed --threads to --n_workers to match Cornell's exact grammar
    cmd = [
        "python", "-m", "birdnet_analyzer.analyze",
        "-o", out_path,
        "--rtype", "csv",
        "--lat", "-1",
        "--lon", "-1",
        "--min_conf", "0.05",
        "--n_workers", "1", 
        wav_path 
    ]
    
    print(f"🚀 Launching Cornell AI engine...", flush=True)
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        print("✅ AI Engine finished processing.", flush=True)
    except subprocess.TimeoutExpired:
        print("❌ AI Engine FROZE and timed out!", flush=True)
        return {"error": "AI Engine timed out. It might be secretly downloading files or ran out of RAM."}
    except Exception as e:
        print(f"❌ AI Engine crashed: {e}", flush=True)
        return {"error": str(e)}

    results = []
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
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
    else:
        print("❌ Result file was never created by the AI!", flush=True)
        return {"error": f"AI Engine Crash: {process.stderr} | {process.stdout}"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🟢 Server booting up on port {port}...", flush=True)
    run(host='0.0.0.0', port=port)
