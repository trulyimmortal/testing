from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import nltk
from typing import List, Dict, Any
import logging
import os
import requests
from functools import lru_cache

# Set HuggingFace token
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuestionCraft AI Backend",
    description="AI-powered question generation using HuggingFace Transformers with authentication",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HuggingFace Configuration
HF_TOKEN = "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"
HF_API_URL = "https://api-inference.huggingface.co"
PRIMARY_MODEL = "valhalla/t5-base-qg-hl"

# Request/Response models
class QuestionGenerationRequest(BaseModel):
    content: str
    max_questions: int = 10
    min_length: int = 10
    max_length: int = 200
    use_hf_api: bool = False  # Option to use HF API instead of local model

class QuestionResponse(BaseModel):
    questions: List[str]
    total_generated: int
    processing_time: float
    model_info: Dict[str, str]
    source: str  # "local" or "huggingface_api"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    hf_token_valid: bool
    message: str

# Global variables for model caching
model = None
tokenizer = None
device = None

@lru_cache(maxsize=1)
def load_model():
    """Load the T5 model and tokenizer with caching"""
    global model, tokenizer, device
    
    try:
        logger.info("Loading T5 question generation model with HF token...")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer with HF token
        model_name = PRIMARY_MODEL
        tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            use_auth_token=HF_TOKEN
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            use_auth_token=HF_TOKEN
        )
        
        # Move model to device
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully with HF authentication!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def validate_hf_token():
    """Validate HuggingFace token"""
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.get(
            f"{HF_API_URL}/models/{PRIMARY_MODEL}",
            headers=headers,
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"HF token validation failed: {str(e)}")
        return False

def generate_questions_hf_api(content: str, max_questions: int = 10) -> List[str]:
    """Generate questions using HuggingFace Inference API"""
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Split content into sentences for better question generation
        sentences = nltk.sent_tokenize(content)[:5]  # Use first 5 sentences
        all_questions = []
        
        for sentence in sentences:
            if len(all_questions) >= max_questions:
                break
                
            payload = {
                "inputs": f"generate question: {sentence}",
                "parameters": {
                    "max_length": 100,
                    "min_length": 10,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_return_sequences": 2
                }
            }
            
            response = requests.post(
                f"{HF_API_URL}/models/{PRIMARY_MODEL}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and "generated_text" in item:
                            question = item["generated_text"].strip()
                            if question and not question.endswith("?"):
                                question += "?"
                            if question and question not in all_questions:
                                all_questions.append(question)
            else:
                logger.warning(f"HF API request failed: {response.status_code}")
        
        return all_questions[:max_questions]
        
    except Exception as e:
        logger.error(f"HF API generation failed: {str(e)}")
        return []

def preprocess_text(text: str) -> str:
    """Clean and preprocess input text"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with generation
    text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
    
    # Ensure text ends with proper punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text

def extract_sentences(text: str, max_sentences: int = 20) -> List[str]:
    """Extract meaningful sentences from text for question generation"""
    try:
        sentences = nltk.sent_tokenize(text)
        
        # Filter sentences by length and content
        filtered_sentences = []
        for sentence in sentences:
            # Skip very short or very long sentences
            if 10 <= len(sentence.split()) <= 50:
                # Skip sentences that are mostly numbers or special characters
                if re.search(r'[a-zA-Z]', sentence):
                    filtered_sentences.append(sentence.strip())
        
        # Return up to max_sentences
        return filtered_sentences[:max_sentences]
        
    except Exception as e:
        logger.error(f"Error extracting sentences: {str(e)}")
        # Fallback: split by periods
        sentences = text.split('.')
        return [s.strip() + '.' for s in sentences if len(s.strip()) > 10][:max_sentences]

def generate_questions_local_model(text: str, max_questions: int = 10) -> List[str]:
    """Generate questions using local T5 model"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Local model not loaded")
    
    try:
        # Preprocess text
        clean_text = preprocess_text(text)
        
        # Extract sentences for question generation
        sentences = extract_sentences(clean_text, max_sentences=15)
        
        if not sentences:
            return ["What are the main concepts discussed in this content?"]
        
        generated_questions = []
        
        # Generate questions from each sentence/context
        for i, sentence in enumerate(sentences):
            if len(generated_questions) >= max_questions:
                break
                
            try:
                # Prepare input for T5 model
                input_text = f"generate question: {sentence}"
                
                # Tokenize input
                inputs = tokenizer.encode(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(device)
                
                # Generate question
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=100,
                        min_length=10,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        no_repeat_ngram_size=2
                    )
                
                # Decode generated question
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the question
                question = question.strip()
                if question and len(question) > 5:
                    # Ensure question ends with question mark
                    if not question.endswith('?'):
                        question += '?'
                    
                    # Capitalize first letter
                    question = question[0].upper() + question[1:] if len(question) > 1 else question.upper()
                    
                    # Avoid duplicates
                    if question not in generated_questions:
                        generated_questions.append(question)
                        
            except Exception as e:
                logger.warning(f"Error generating question from sentence {i}: {str(e)}")
                continue
        
        # If we don't have enough questions, generate some generic ones
        if len(generated_questions) < 3:
            generic_questions = generate_generic_questions(clean_text)
            generated_questions.extend(generic_questions)
        
        # Remove duplicates and return
        unique_questions = list(dict.fromkeys(generated_questions))
        return unique_questions[:max_questions]
        
    except Exception as e:
        logger.error(f"Error in local question generation: {str(e)}")
        # Fallback to generic questions
        return generate_generic_questions(text)

def generate_generic_questions(text: str) -> List[str]:
    """Generate generic questions when model fails"""
    # Extract key terms from text
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    key_terms = list(set(words))[:5]
    
    generic_questions = [
        "What are the main concepts discussed in this content?",
        "Explain the key principles mentioned in the text.",
        "How do the different elements relate to each other?",
        "What is the significance of the topics covered?",
        "Describe the important aspects highlighted in the material."
    ]
    
    # Add term-specific questions if we found key terms
    for term in key_terms[:3]:
        generic_questions.append(f"What is {term} and why is it important?")
        generic_questions.append(f"How does {term} relate to the overall topic?")
    
    return generic_questions[:10]

@app.on_event("startup")
async def startup_event():
    """Load model and validate HF token on startup"""
    logger.info("Starting QuestionCraft AI Backend with HuggingFace integration...")
    
    # Validate HF token
    token_valid = validate_hf_token()
    if token_valid:
        logger.info("HuggingFace token validated successfully!")
    else:
        logger.warning("HuggingFace token validation failed")
    
    # Load local model
    success = load_model()
    if not success:
        logger.error("Failed to load local model on startup")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    hf_token_valid = validate_hf_token()
    
    status = "healthy" if (model_loaded or hf_token_valid) else "unhealthy"
    message = "QuestionCraft AI Backend is running"
    
    if model_loaded and hf_token_valid:
        message += " (Local model + HF API available)"
    elif model_loaded:
        message += " (Local model only)"
    elif hf_token_valid:
        message += " (HF API only)"
    else:
        message += " (Limited functionality)"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        hf_token_valid=hf_token_valid,
        message=message
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    model_loaded = model is not None and tokenizer is not None
    hf_token_valid = validate_hf_token()
    
    return HealthResponse(
        status="healthy" if (model_loaded or hf_token_valid) else "unhealthy",
        model_loaded=model_loaded,
        hf_token_valid=hf_token_valid,
        message="All systems operational" if (model_loaded or hf_token_valid) else "Service degraded"
    )

@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(request: QuestionGenerationRequest):
    """Generate questions from input content using best available method"""
    import time
    start_time = time.time()
    
    try:
        # Validate input
        if not request.content or len(request.content.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Content must be at least 10 characters long"
            )
        
        questions = []
        source = "unknown"
        model_info = {"model_name": "unknown", "device": "unknown", "framework": "unknown"}
        
        # Try HuggingFace API first if requested or if local model not available
        if request.use_hf_api or model is None:
            try:
                logger.info("Attempting HuggingFace API generation...")
                questions = generate_questions_hf_api(request.content, request.max_questions)
                if questions:
                    source = "huggingface_api"
                    model_info = {
                        "model_name": PRIMARY_MODEL,
                        "device": "huggingface_cloud",
                        "framework": "transformers_api"
                    }
                    logger.info(f"HF API generated {len(questions)} questions")
            except Exception as e:
                logger.warning(f"HF API failed: {str(e)}")
        
        # Fallback to local model if HF API failed or not requested
        if not questions and model is not None:
            try:
                logger.info("Using local model generation...")
                questions = generate_questions_local_model(request.content, request.max_questions)
                source = "local_model"
                model_info = {
                    "model_name": PRIMARY_MODEL,
                    "device": str(device) if device else "cpu",
                    "framework": "transformers_local"
                }
                logger.info(f"Local model generated {len(questions)} questions")
            except Exception as e:
                logger.warning(f"Local model failed: {str(e)}")
        
        # Final fallback to generic questions
        if not questions:
            logger.info("Using generic question generation...")
            questions = generate_generic_questions(request.content)
            source = "generic_fallback"
            model_info = {
                "model_name": "rule_based",
                "device": "local",
                "framework": "custom"
            }
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {len(questions)} questions in {processing_time:.2f}s using {source}")
        
        return QuestionResponse(
            questions=questions,
            total_generated=len(questions),
            processing_time=round(processing_time, 2),
            model_info=model_info,
            source=source
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_questions: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/analyze-content")
async def analyze_content(request: QuestionGenerationRequest):
    """Analyze content and return metadata along with questions"""
    try:
        # Generate questions
        questions_response = await generate_questions(request)
        
        # Analyze content for additional metadata
        content = request.content
        word_count = len(content.split())
        sentence_count = len(nltk.sent_tokenize(content))
        
        # Extract potential topics/keywords
        words = re.findall(r'\b[A-Z][a-z]+\b', content)
        topics = list(set(words))[:10]
        
        return {
            "questions": questions_response.questions,
            "metadata": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "estimated_reading_time": round(word_count / 200, 1),  # Average reading speed
                "potential_topics": topics,
                "content_length": len(content),
                "processing_time": questions_response.processing_time,
                "source": questions_response.source
            },
            "model_info": questions_response.model_info
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
