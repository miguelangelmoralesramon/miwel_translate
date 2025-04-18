from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import os
import sys
from typing import Dict, List

# Add the directory containing transformer.py to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Transformer model and required functions
from transformer import Transformer, translate

app = FastAPI(title="Miwel Translate API", 
              description="API for English to Spanish translation using a custom Transformer model",
              version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class TranslationInput(BaseModel):
    sentence: str = Field(..., example="hello world", min_length=1)

class TranslationOutput(BaseModel):
    translation: str

# Constants for tokenization and model configuration
START_TOKEN = '<SOE>'
PADDING_TOKEN = '<PAD>'
END_TOKEN = '<EOE>'

# Create vocabularies for English and Spanish
english_vocab = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                ':', '<', '=', '>', '?', '@',
                '[', '\\', ']', '^', '_', '`',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                'y', 'z',
                '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

spanish_vocab = [START_TOKEN, ' ','¡', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                ':', '<', '=', '>', '¿','?', '@',
                '[', '\\', ']', '^', '_', '`',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n','ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                'y', 'z','á','é','í','ó','ú',
                '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

# Create mappings between indices and tokens
ind_to_esp = {k:v for k,v in enumerate(spanish_vocab)}
esp_to_ind = {v:k for k,v in enumerate(spanish_vocab)}
ind_to_eng = {k:v for k,v in enumerate(english_vocab)}
eng_to_ind = {v:k for k,v in enumerate(english_vocab)}

# Model parameters
d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 4
max_sequence_length = 300
esp_vocab_size = len(spanish_vocab)

# Function to check if a sentence has valid tokens
def is_valid_tokens(sentence: str, vocab: List[str]) -> bool:
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

# Function to check if a sentence is within the max length
def is_valid_length(sentence: str, max_length: int) -> bool:
    return len(list(sentence)) < (max_length - 1)

# Load the model at startup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = None

@app.on_event("startup")
async def startup_event():
    global model
    
    # Initialize the transformer model
    model = Transformer(
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        num_heads=num_heads,
        drop_prob=drop_prob,
        num_layers=num_layers,
        max_sequence_length=max_sequence_length,
        kn_vocab_size=esp_vocab_size,
        english_to_index=eng_to_ind,
        kannada_to_index=esp_to_ind,
        START_TOKEN=START_TOKEN,
        END_TOKEN=END_TOKEN,
        PADDING_TOKEN=PADDING_TOKEN
    )
    
    # Load the pre-trained model weights
    try:
        # Check if model file exists
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "models", "transformer_final.pth.tar")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            print("Checking alternative locations...")
            
            # Try alternative locations
            alt_paths = [
                "./transformer_final.pth.tar",
                "./models/transformer_final.pth.tar",
                "../transformer_final.pth.tar"
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Found model at {model_path}")
                    break
            else:
                raise FileNotFoundError("Could not find model file")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get("/health")
async def health_check() -> Dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": str(device)}

@app.post("/translate", response_model=TranslationOutput)
async def translate_text(input_data: TranslationInput) -> Dict[str, str]:
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get the input sentence and convert to lowercase
    sentence = input_data.sentence.lower()
    
    # Validate the input
    if not sentence.strip():
        raise HTTPException(status_code=400, detail="Empty sentence")
    
    if not is_valid_tokens(sentence, english_vocab):
        raise HTTPException(status_code=400, 
                           detail="Sentence contains invalid characters. Only lowercase English letters, numbers, and basic punctuation are supported.")
    
    if not is_valid_length(sentence, max_sequence_length):
        raise HTTPException(status_code=400, 
                           detail=f"Sentence is too long. Maximum length is {max_sequence_length-1} characters.")
    
    # Perform the translation
    try:
        output = translate(model, sentence)
        
        # Remove the start and end tokens if present
        if output.startswith(START_TOKEN):
            output = output[len(START_TOKEN):]
        if END_TOKEN in output:
            output = output[:output.index(END_TOKEN)]
        
        return {"translation": output.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)