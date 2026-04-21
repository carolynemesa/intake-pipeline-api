from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import joblib
import re

app = FastAPI()

# Load models once at startup
nlp_ner = spacy.load("intake_ner_model")
clf_vertical = joblib.load("clf_vertical.pkl")
clf_maturity = joblib.load("clf_maturity.pkl")

def clean_transcript(text):
    # Remove speaker labels e.g. "John:" or "SPEAKER 1:"
    text = re.sub(r'^[A-Z][^:]{0,30}:\s*', '', text, flags=re.MULTILINE)
    # Remove filler words
    text = re.sub(r'\b(um|uh|like|you know|I mean|sort of|kind of)\b', 
                  '', text, flags=re.IGNORECASE)
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class IntakeRequest(BaseModel):
    text: str
    is_transcript: bool = False  # set True for meeting transcripts

@app.get("/")
def health_check():
    return {"status": "online", "model": "intake-pipeline-api"}

@app.post("/extract")
def extract(req: IntakeRequest):
    text = clean_transcript(req.text) if req.is_transcript else req.text

    # NER
    doc = nlp_ner(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)

    # Classification
    vertical = clf_vertical.predict([text])[0]
    maturity = clf_maturity.predict([text])[0]

    return {
        "entities": entities,
        "industry_vertical": vertical,
        "maturity_tier": maturity,
        "cleaned_text": text if req.is_transcript else None
    }