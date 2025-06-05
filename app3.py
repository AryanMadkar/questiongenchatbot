from flask import Flask, render_template, request, jsonify, Response
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
import torch
import os
import time
import json
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
from threading import Thread
from flask import send_from_directory

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using: {'GPU' if device == 'cuda' else 'CPU'}")

# Get current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === MODEL CONFIG ===
MODEL_CONFIG = {
    "llama-3": {
        "local_path": os.path.join(BASE_DIR, "llama-3"),
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "template": "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    },
    "mistral": {
        "local_path": os.path.join(BASE_DIR, "mistral"),
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "template": "<s>[INST] {prompt} [/INST]"
    },
    "phi-2": {
        "local_path": os.path.join(BASE_DIR, "phi-2"),
        "hf_id": "microsoft/phi-2",
        "template": "Instruct: {prompt}\nOutput:\n"
    }
}
loaded_models = {}
loaded_chains = {}
loaded_models_by_path = {}  # Cache models by their local path

def download_model(model_key):
    """Download model from Hugging Face Hub if not present locally"""
    config = MODEL_CONFIG.get(model_key)
    if not config:
        print(f"❌ Unknown model: {model_key}")
        return False
        
    local_path = config["local_path"]
    hf_id = config["hf_id"]
    
    # Check if model directory exists and has required files
    if os.path.exists(local_path):
        required_files = ["config.json", "model.safetensors"]
        # Check if required files exist
        required_exist = all(os.path.exists(os.path.join(local_path, f)) for f in required_files)
        
        if required_exist:
            print(f"✅ Model found locally: {model_key}")
            return True
        else:
            print(f"⚠️ Partial download detected for {model_key}, re-downloading...")
            shutil.rmtree(local_path)
    
    print(f"⬇️ Downloading model: {model_key} from Hugging Face Hub...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)
        
        # Download model files
        snapshot_download(
            repo_id=hf_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            revision="main"
        )
        
        print(f"✅ Successfully downloaded {model_key}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        # Clean up partial downloads
        if os.path.exists(local_path):
            try:
                shutil.rmtree(local_path)
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Could not clean up {local_path}: {cleanup_error}")
        return False

def load_model(model_key):
    """Load model weights and tokenizer"""
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_key}")
    
    config = MODEL_CONFIG[model_key]
    local_path = config["local_path"]
    
    # Check if model is already loaded by path
    if local_path in loaded_models_by_path:
        print(f"♻️ Using cached model by path: {local_path}")
        return loaded_models_by_path[local_path]
    
    # Ensure model is downloaded
    if not download_model(model_key):
        raise Exception(f"Model {model_key} not available and download failed")
    
    print(f"📦 Loading model: {model_key}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Move model to device if not using device_map
        if device == "cpu":
            model = model.to(device)

        # Cache the loaded components by path
        loaded_models_by_path[local_path] = (model, tokenizer)
        print(f"✅ Model loaded successfully: {model_key}")
        return model, tokenizer
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Clear cache to force reload on next attempt
        if local_path in loaded_models_by_path:
            del loaded_models_by_path[local_path]
        raise Exception(f"Failed to load model {model_key}: {str(e)}")

def get_llm_chain(model_key, temperature=0.7, top_p=0.95, max_tokens=200):
    """Create LangChain pipeline with generation parameters"""
    # Generate cache key based on parameters
    cache_key = f"{model_key}-{temperature}-{top_p}-{max_tokens}"
    
    if cache_key in loaded_chains:
        print(f"♻️ Using cached chain: {cache_key}")
        return loaded_chains[cache_key]
    
    # Get model components
    model, tokenizer = load_model(model_key)
    
    # Create text generation pipeline
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        device=0 if device == "cuda" else -1,
        return_full_text=False
    )

    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    
    # Create prompt template
    template = MODEL_CONFIG[model_key]["template"]
    prompt = PromptTemplate.from_template(template)
    
    # Create LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Cache the chain
    loaded_chains[cache_key] = chain
    return chain

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/models", methods=["GET"])
def list_models():
    """Endpoint to list available models and their status"""
    models_info = []
    for name, config in MODEL_CONFIG.items():
        local_path = config["local_path"]
        exists = os.path.exists(local_path)
        
        # Check if model has required files
        ready = False
        size = "N/A"
        
        if exists:
            try:
                files = os.listdir(local_path)
                required_files = ["config.json", "model.safetensors"]
                ready = all(f in files for f in required_files)
                
                # Calculate approximate size
                if ready:
                    total_size = sum(os.path.getsize(os.path.join(local_path, f)) 
                                 for f in files if os.path.isfile(os.path.join(local_path, f)))
                    size = f"{total_size / (1024**3):.1f} GB"
            except Exception as e:
                print(f"Error checking {name}: {e}")
                ready = False
        
        # Check if model is loaded in memory
        loaded_in_memory = local_path in loaded_models_by_path
        
        models_info.append({
            "name": name,
            "downloaded": ready,
            "path": local_path,
            "size": size,
            "loaded": loaded_in_memory
        })
    
    return jsonify(models_info)

@app.route("/generate", methods=["POST"])
def generate():
    """Standard text generation endpoint using LangChain"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Empty prompt provided"}), 400
            
        model_key = data.get("model", "llama-3")
        temperature = max(0.1, min(2.0, float(data.get("temperature", 0.7))))
        top_p = max(0.1, min(1.0, float(data.get("top_p", 0.95))))
        max_tokens = max(1, min(1000, int(data.get("max_tokens", 200))))

        # Get LangChain pipeline
        chain = get_llm_chain(model_key, temperature, top_p, max_tokens)
        
        # Generate response
        start_time = time.time()
        response = chain.invoke({"prompt": prompt})["text"]
        duration = time.time() - start_time

        return jsonify({
            "response": response.strip(),
            "time": round(duration, 2),
            "model": model_key,
            "tokens": len(chain.llm.pipeline.tokenizer.encode(response))
        })
    
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route("/stream", methods=["POST"])
def stream():
    """Streaming text generation endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Empty prompt provided"}), 400
            
        model_key = data.get("model", "llama-3")
        temperature = max(0.1, min(2.0, float(data.get("temperature", 0.7))))
        top_p = max(0.1, min(1.0, float(data.get("top_p", 0.95))))
        max_tokens = max(1, min(1000, int(data.get("max_tokens", 200))))

        # Load model components
        model, tokenizer = load_model(model_key)
        model.eval()  # Set to evaluation mode
        
        # Format prompt with template
        template = MODEL_CONFIG[model_key]["template"]
        formatted_prompt = template.format(prompt=prompt)
        
        # Prepare inputs
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300  # Increase timeout for long generations
        )

        # Generation parameters
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        # Start generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def generate_stream():
            # Stream tokens as they come
            for new_token in streamer:
                if new_token:
                    yield f"data: {json.dumps({'token': new_token})}\n\n"
            yield "data: {\"done\": true}\n\n"

        return Response(
            generate_stream(), 
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # Disable buffering for nginx
            }
        )
    
    except Exception as e:
        print(f"Stream setup error: {e}")
        return jsonify({"error": f"Stream failed: {str(e)}"}), 500

@app.route("/unload", methods=["POST"])
def unload_model():
    """Endpoint to unload models and free memory"""
    try:
        data = request.json
        model_key = data.get("model") if data else None
        
        if not model_key or model_key not in MODEL_CONFIG:
            return jsonify({"error": "Invalid model specified"}), 400
            
        config = MODEL_CONFIG[model_key]
        local_path = config["local_path"]
            
        # Remove chains associated with this model
        chains_to_remove = [k for k in loaded_chains if k.startswith(model_key)]
        for k in chains_to_remove:
            del loaded_chains[k]
            
        # Unload model if it exists in cache
        if local_path in loaded_models_by_path:
            del loaded_models_by_path[local_path]
            print(f"🗑️ Unloaded model: {model_key} from path {local_path}")
            
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Cleared GPU memory")
            
        return jsonify({
            "message": f"Model {model_key} unloaded successfully",
            "unloaded_chains": len(chains_to_remove)
        })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        # Cleanup on exit
        loaded_models_by_path.clear()
        loaded_chains.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Cleanup completed")