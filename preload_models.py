import os

# Limit CPU threads during pre-load
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

print("⬇️ Pre-downloading BirdNET AI models and taxonomy files to disk cache...", flush=True)

try:
    from birdnet_analyzer import model, species
    
    # 1. Download base acoustic neural network model
    print("  -> Pre-loading acoustic model...", flush=True)
    model.load_model()
    
    # 2. Download geo-location ONNX model & taxonomy (using sample EU coordinates)
    print("  -> Pre-loading geo-location model...", flush=True)
    species.get_species_list(50.85, 4.35, 0.15)
    
    print("✅ All BirdNET models successfully downloaded and cached on disk!", flush=True)
except Exception as e:
    print(f"⚠️ Pre-load notification: {e}", flush=True)
