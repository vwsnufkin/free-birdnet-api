@route('/analyze', method=['OPTIONS', 'POST'])
def analyze_audio():
    # Keep these at the top for the OPTIONS handshake
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization'
    
    if request.method == 'OPTIONS':
        response.status = 200
        return {}

    # ... [Your exact ffmpeg conversion and AI processing code stays here] ...

    # 🚨 SYSTEM REPAIR: Re-inject CORS keys right before returning data to Lovable
    if "Columns must be same length as key" in process.stderr:
        print("✅ Cornell empty-result bug caught. Returning 0 birds.", flush=True)
        response.headers['Access-Control-Allow-Origin'] = '*' # RE-APPLY HERE
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
            
            # 🚨 RE-APPLY RIGHT BEFORE SUCCESS RETURN
            response.headers['Access-Control-Allow-Origin'] = '*' 
            return {"results": results}
        
    print("❌ Result file was never created by the AI!", flush=True)
    # 🚨 RE-APPLY RIGHT BEFORE ERROR RETURN
    response.headers['Access-Control-Allow-Origin'] = '*' 
    return {"error": f"AI Engine Crash: {process.stderr} | {process.stdout}"}
