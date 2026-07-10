import os
import subprocess
import csv
from bottle import route, run, request, response

@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio():
    # 1. Bypass browser security blocks (CORS)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    
    if request.method == 'OPTIONS':
        return {}

    # 2. Receive the audio from your website
    upload = request.files.get('audio')
    
    if not upload:
        response.status = 400
        return {"error": "No audio file provided"}

    audio_path = '/tmp/recording.webm'
    out_path = '/tmp/result.csv'
    
    if os.path.exists(audio_path): os.remove(audio_path)
    if os.path.exists(out_path): os.remove(out_path)
        
    upload.save(audio_path)

    # 3. Use the new modular command structure to trigger the AI
    cmd = [
        "python", "-m", "birdnet_analyzer.analyze",
        "--i", audio_path,
        "--o", out_path,
        "--rtype", "csv",
        "--lat", "-1",
        "--lon", "-1",
        "--min_conf", "0.05"
    ]
    subprocess.run(cmd)

    # 4. Read Cornell's results and send them back to your website
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
        
        # Sort by the highest confidence match
        results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    return {"results": results}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    run(host='0.0.0.0', port=port)
