from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import requests
import os
import time
import json
from threading import Thread
import hashlib
import logging
from datetime import datetime
from functools import wraps
import re
import threading
from collections import defaultdict
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Set this environment variable

# Alternative models you can use (just change the URL):
# "https://api-inference.huggingface.co/models/microsoft/phi-2"
# "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
# "https://api-inference.huggingface.co/models/google/flan-t5-large"

if not HF_API_TOKEN:
    logger.warning("HUGGINGFACE_API_TOKEN not set. API calls may be rate-limited.")

def call_huggingface_api(prompt, max_tokens=500, temperature=0.7, top_p=0.95):
    """
    Call Hugging Face Inference API
    """
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    # For text generation models
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                estimated_time = response.json().get("estimated_time", 20)
                logger.info(f"Model loading, waiting {estimated_time} seconds...")
                time.sleep(min(estimated_time, 60))
                continue
                
            elif response.status_code == 429:
                # Rate limited
                logger.warning("Rate limited, waiting 60 seconds...")
                time.sleep(60)
                continue
                
            elif response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                else:
                    return str(result).strip()
                    
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                if attempt == max_retries - 1:
                    raise Exception(f"API call failed: {response.status_code}")
                    
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise Exception("Request timed out")
        except Exception as e:
            logger.error(f"API call error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise e
        
        time.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception("Failed to get response from Hugging Face API")

def stream_huggingface_api(prompt, max_tokens=500, temperature=0.7, top_p=0.95):
    """
    Stream response from Hugging Face API (simulated streaming)
    Note: HF Inference API doesn't support true streaming for free tier
    """
    try:
        response = call_huggingface_api(prompt, max_tokens, temperature, top_p)
        
        # Simulate streaming by yielding words
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.05)  # Small delay to simulate streaming
            
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"Error: {str(e)}"

# Enhanced MCQ Generation Functionality (keeping your existing cache and rate limiting)
class AdvancedCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                if time.time() - self.creation_times[key] > self.ttl_seconds:
                    self._remove(key)
                    return None
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.creation_times[key] = time.time()
    
    def _remove(self, key):
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.creation_times[key]
    
    def _evict_lru(self):
        if not self.cache:
            return
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier):
        with self.lock:
            now = time.time()
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.window_seconds
            ]
            
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                return True
            return False

# Initialize systems
cache = AdvancedCache(max_size=2000, ttl_seconds=7200)
rate_limiter = RateLimiter(max_requests=30, window_seconds=3600)  # Reduced for API limits

PROMPT_TEMPLATES = {
    "academic": """
You are an expert educational content creator and a problem solver who is very sarcastic and a bit harsh while talking but always helps the student. Generate {num_questions} academically rigorous multiple-choice questions about "{topic}" at {difficulty} level.

Requirements:
- Questions must test deep understanding, not just memorization
- Include application, analysis, and synthesis level questions
- Options should be plausible and challenging
- Avoid obvious incorrect answers
- Include explanations for correct answers

Difficulty Guidelines:
- Easy: Basic concepts and definitions
- Medium: Application and analysis
- Hard: Synthesis, evaluation, and complex problem-solving
- Expert: Advanced theoretical concepts and real-world applications

Return ONLY valid JSON in this exact structure:
{{
  "metadata": {{
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "total_questions": {num_questions},
    "generation_time": "{timestamp}"
  }},
  "questions": [
    {{
      "id": 1,
      "question": "Clear, specific question text",
      "options": {{
        "A": "First option",
        "B": "Second option", 
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct_answer": "A",
      "explanation": "Detailed explanation of why this answer is correct",
      "bloom_level": "analyze",
      "estimated_time_seconds": 45,
      "tags": ["concept1", "concept2"]
    }}
  ]
}}
""",
    
    "practical": """
You are an expert educational content creator. Generate {num_questions} practical, scenario-based multiple-choice questions about "{topic}" at {difficulty} level.

Focus on:
- Real-world applications and case studies
- Problem-solving scenarios
- Best practices and common mistakes
- Industry standards and procedures

Return ONLY valid JSON in the same structure as provided above.
""",
    
    "conceptual": """
You are an expert educational content creator. Generate {num_questions} conceptual multiple-choice questions about "{topic}" at {difficulty} level.

Focus on:
- Theoretical understanding
- Relationships between concepts
- Cause and effect relationships
- Comparative analysis

Return ONLY valid JSON in the same structure as provided above.
"""
}

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)
            
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": 3600
            }), 429
        return f(*args, **kwargs)
    return decorated_function

def validate_input(data: dict) -> tuple:
    """Enhanced input validation"""
    if not data:
        return False, "No data provided"
    
    topic = data.get("topic", "").strip()
    if not topic or len(topic) < 2:
        return False, "Topic must be at least 2 characters long"
    
    if len(topic) > 200:
        return False, "Topic must be less than 200 characters"
    
    difficulty = data.get("difficulty", "medium").lower()
    valid_difficulties = ["easy", "medium", "hard", "expert"]
    if difficulty not in valid_difficulties:
        return False, f"Difficulty must be one of: {', '.join(valid_difficulties)}"
    
    num_questions = data.get("num_questions", 5)
    if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 20:  # Reduced max for API limits
        return False, "Number of questions must be between 1 and 20"
    
    question_type = data.get("question_type", "academic")
    valid_types = ["academic", "practical", "conceptual"]
    if question_type not in valid_types:
        return False, f"Question type must be one of: {', '.join(valid_types)}"
    
    return True, ""

def generate_cache_key(topic: str, difficulty: str, num_questions: int, question_type: str) -> str:
    """Generate a secure cache key"""
    content = f"{topic.lower()}_{difficulty.lower()}_{num_questions}_{question_type}"
    return hashlib.md5(content.encode()).hexdigest()

def build_enhanced_prompt(topic: str, difficulty: str, num_questions: int, question_type: str) -> str:
    """Build enhanced prompt based on question type"""
    timestamp = datetime.now().isoformat()
    template = PROMPT_TEMPLATES.get(question_type, PROMPT_TEMPLATES["academic"])
    
    return template.format(
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
        timestamp=timestamp
    )

def validate_generated_content(content: str) -> tuple:
    """Validate and parse generated content"""
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return False, None
        
        json_data = json.loads(json_match.group())
        
        required_fields = ["metadata", "questions"]
        if not all(field in json_data for field in required_fields):
            return False, None
        
        questions = json_data.get("questions", [])
        if not questions:
            return False, None
        
        for i, q in enumerate(questions):
            required_q_fields = ["question", "options", "correct_answer"]
            if not all(field in q for field in required_q_fields):
                logger.warning(f"Question {i+1} missing required fields")
                return False, None
            
            options = q.get("options", {})
            if len(options) != 4 or not all(key in options for key in ["A", "B", "C", "D"]):
                logger.warning(f"Question {i+1} has invalid options structure")
                return False, None
            
            if q.get("correct_answer") not in ["A", "B", "C", "D"]:
                logger.warning(f"Question {i+1} has invalid correct answer")
                return False, None
        
        return True, json_data
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Content validation error: {e}")
        return False, None

def enhance_response(json_data: dict) -> dict:
    """Add enhancements to the response"""
    questions = json_data.get("questions", [])
    
    total_time = sum(q.get("estimated_time_seconds", 60) for q in questions)
    
    bloom_levels = [q.get("bloom_level", "remember") for q in questions]
    bloom_distribution = {level: bloom_levels.count(level) for level in set(bloom_levels)}
    
    json_data["analytics"] = {
        "total_estimated_time_minutes": round(total_time / 60, 1),
        "average_time_per_question": round(total_time / len(questions), 1),
        "bloom_taxonomy_distribution": bloom_distribution,
        "difficulty_score": calculate_difficulty_score(json_data.get("metadata", {}).get("difficulty", "medium")),
        "quality_indicators": {
            "has_explanations": all("explanation" in q for q in questions),
            "has_bloom_levels": all("bloom_level" in q for q in questions),
            "has_tags": all("tags" in q for q in questions)
        }
    }
    
    return json_data

def calculate_difficulty_score(difficulty: str) -> float:
    scores = {"easy": 0.25, "medium": 0.5, "hard": 0.75, "expert": 1.0}
    return scores.get(difficulty.lower(), 0.5)



@app.route("/models", methods=["GET"])
def list_models():
    """Return information about the Hugging Face API model"""
    return jsonify([{
        "name": "huggingface-api",
        "downloaded": True,
        "path": "Hugging Face Inference API",
        "size": "Cloud-based",
        "loaded": True,
        "model_url": HF_API_URL
    }])

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Empty prompt provided"}), 400
            
        temperature = max(0.1, min(2.0, float(data.get("temperature", 0.7))))
        top_p = max(0.1, min(1.0, float(data.get("top_p", 0.95))))
        max_tokens = max(1, min(500, int(data.get("max_tokens", 200))))  # Reduced for API limits

        start_time = time.time()
        response = call_huggingface_api(prompt, max_tokens, temperature, top_p)
        duration = time.time() - start_time

        return jsonify({
            "response": response,
            "time": round(duration, 2),
            "model": "huggingface-api",
            "tokens": len(response.split())  # Approximate token count
        })
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route("/stream", methods=["POST"])
def stream():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Empty prompt provided"}), 400
            
        temperature = max(0.1, min(2.0, float(data.get("temperature", 0.7))))
        top_p = max(0.1, min(1.0, float(data.get("top_p", 0.95))))
        max_tokens = max(1, min(500, int(data.get("max_tokens", 200))))

        def generate_stream():
            try:
                for token in stream_huggingface_api(prompt, max_tokens, temperature, top_p):
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: {\"done\": true}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            generate_stream(), 
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
    
    except Exception as e:
        logger.error(f"Stream setup error: {e}")
        return jsonify({"error": f"Stream failed: {str(e)}"}), 500

@app.route("/unload", methods=["POST"])
def unload_model():
    """Placeholder for API compatibility - no local model to unload"""
    return jsonify({
        "message": "Using Hugging Face API - no local model to unload",
        "unloaded_chains": 0
    })

@app.route('/generate_mcqs', methods=['POST', 'OPTIONS'])
@rate_limit
def generate_mcqs():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    if not request.is_json:
        return jsonify({
            "error": "Unsupported Media Type",
            "message": "Content-Type must be application/json"
        }), 415
    
    try:
        data = request.get_json()
        
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({"error": "Validation failed", "message": error_msg}), 400
        
        topic = data["topic"].strip()
        difficulty = data.get("difficulty", "medium").lower()
        num_questions = data.get("num_questions", 5)
        question_type = data.get("question_type", "academic")
        
        cache_key = generate_cache_key(topic, difficulty, num_questions, question_type)
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for topic: {topic}")
            return jsonify({**cached_result, "cached": True})
        
        prompt = build_enhanced_prompt(topic, difficulty, num_questions, question_type)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating MCQs (attempt {attempt + 1}) - Topic: {topic}")
                
                response = call_huggingface_api(prompt, max_tokens=1000, temperature=0.2, top_p=0.9)
                
                is_valid_content, json_data = validate_generated_content(response)
                if is_valid_content and json_data:
                    enhanced_data = enhance_response(json_data)
                    cache.set(cache_key, enhanced_data)
                    logger.info(f"Generated {len(enhanced_data.get('questions', []))} questions")
                    return jsonify({**enhanced_data, "cached": False})
                
                logger.warning(f"Invalid content generated on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)
        
        return jsonify({
            "error": "Generation failed",
            "message": "Unable to generate valid questions after multiple attempts"
        }), 500
        
    except Exception as e:
        logger.error(f"MCQ generation error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_service": "Hugging Face Inference API",
        "cache_size": len(cache.cache)
    })

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting MCQ Generator API with Hugging Face Integration...")
    logger.info(f"API URL: {HF_API_URL}")
    logger.info(f"Cache configured: max_size={cache.max_size}")
    logger.info(f"Rate limiting: {rate_limiter.max_requests} requests per hour")
    
    if not HF_API_TOKEN:
        logger.warning("Consider setting HUGGINGFACE_API_TOKEN environment variable for higher rate limits")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
