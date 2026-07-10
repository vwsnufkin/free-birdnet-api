import os
import subprocess
import csv
from bottle import route, run, request, response

@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    
    if request.method == 'OPTIONS':
        return {}

    upload = request.files.get('audio')
    if not upload:
        return {"error": "No audio file provided"}

    # Save the raw messy file, and force-convert it to a clean WAV
    raw_path = '/tmp/raw_upload'
    wav_path = '/tmp/recording.wav'
    out_path = '/tmp/result.csv'
    
    for p in [raw_path, wav_path, out_path]:
        if os.path.exists(p): os.remove(p)
        
    upload.save(raw_path)

    # 1. Force convert ANY browser audio format into a pristine 48kHz WAV file
    try:
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "48000", wav_path], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return {"error": f"Audio Conversion Failed. Log: {e.stderr.decode()}"}

    # 2. Run the AI! Notice how the input file is at the VERY END, matching their exact syntax.
    cmd = [
        "python", "-m", "birdnet_analyzer.analyze",
        "-o", out_path,
        "--rtype", "csv",
        "--lat", "-1",
        "--lon", "-1",
        "--min_conf", "0.05",
        wav_path 
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)

    # 3. Read the results
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
        return {"results": results}
    else:
        return {"error": f"AI Engine Crash: {process.stderr} | {process.stdout}"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    run(host='0.0.0.0', port=port)
