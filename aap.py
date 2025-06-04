from flask import Flask, render_template, request, jsonify, session
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_PATH = "./TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Using device: {'GPU' if device == 0 else 'CPU'}")

# === Load model if exists, else download ===
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Local model not found. Downloading Phi-2...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
    else:
        print("✅ Local model found. Loading from disk...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == 0 else torch.float32,
        device_map="auto" if device == 0 else None
    )
    
    return tokenizer, model

# === Init pipeline ===
tokenizer, model = load_model()
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15,
    device=device
)

# === Prompt wrapper ===
def generate_response(prompt):
    start = time.time()
    output = pipe(prompt, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    response = output[len(prompt):].strip()
    print(f"⏱️ Response time: {time.time() - start:.2f}s")
    return response

# === Init session memory ===
@app.before_request
def init_session():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "session_start" not in session:
        session["session_start"] = time.strftime("%Y-%m-%d %H:%M:%S")

# === Routes ===
@app.route("/")
def home():
    return render_template("index.html")  # add your frontend

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()
    
    if not user_input:
        return jsonify({"response": "Say something first, bruh."})

    history = "\n".join([f"User: {h['user']}\nAI: {h['bot']}" for h in session["chat_history"]])
    prompt = f"{history}\nUser: {user_input}\nAI:"
    response = generate_response(prompt)

    session["chat_history"].append({"user": user_input, "bot": response})

    return jsonify({
        "response": response,
        "chat_history": session["chat_history"]
    })

@app.route("/reset", methods=["POST"])
def reset():
    session.pop("chat_history", None)
    session.pop("session_start", None)
    return jsonify({"status": "Reset done!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
