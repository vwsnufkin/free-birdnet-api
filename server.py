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
        return {"error": "Server is currently busy analyzing audio."}

    upload = request.files.get('audio') or request.files.get('file')
    if not upload:
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": "No audio file provided"}

    req_id = str(uuid.uuid4())[:8]
    raw_path = f'/tmp/raw_{req_id}'
    wav_path = f'/tmp/rec_{req_id}.wav'

    # Optional: Log incoming location data if passed in form fields
    lat = request.forms.get('lat') or request.forms.get('latitude')
    lon = request.forms.get('lon') or request.forms.get('longitude')
    
    print(f"📡 [{req_id}] Incoming audio request received!", flush=True)
    if lat and lon:
        print(f"🌍 [{req_id}] GPS Location received: Lat {lat}, Lon {lon}", flush=True)

    try:
        IS_BUSY = True
        upload.save(raw_path)
        print(f"✅ [{req_id}] Audio file saved. Converting format...", flush=True)

        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path, 
            "-filter:a", "volume=10dB", 
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", 
            wav_path
        ], check=True, capture_output=True)

        print(f"✅ [{req_id}] Audio successfully converted.", flush=True)

        load_labels()
        interpreter = init_tflite_interpreter()

        if not interpreter:
            response.headers['Access-Control-Allow-Origin'] = '*'
            return {"error": "TFLite Model interpreter failed to initialize"}

        rate, data = wav.read(wav_path)
        sig = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)

        min_samples = 144000
        if len(sig) < min_samples:
            sig = np.pad(sig, (0, min_samples - len(sig)))

        chunks = [sig[i:i + min_samples] for i in range(0, len(sig) - min_samples + 1, min_samples)]
        if not chunks:
            chunks.append(sig[:min_samples])

        results_map = {}

        for chunk in chunks:
            in_data = np.expand_dims(chunk, axis=0).astype(np.float32)
            interpreter.set_tensor(INPUT_DETAILS[0]['index'], in_data)
            interpreter.invoke()
            scores = interpreter.get_tensor(OUTPUT_DETAILS[0]['index'])[0]
            
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
        print(f"🎯 [{req_id}] Classification complete. Identified {len(formatted_results)} species.", flush=True)

        response.headers['Access-Control-Allow-Origin'] = '*' 
        return {"results": formatted_results}

    except Exception as e:
        print(f"❌ [{req_id}] Processing error: {str(e)}", flush=True)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"error": str(e)}

    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)
        IS_BUSY = False
        gc.collect()
