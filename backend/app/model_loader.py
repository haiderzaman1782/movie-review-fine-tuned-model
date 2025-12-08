from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
import torch
import os

class SentimentModel:
    def __init__(self):
        self.pipe = None
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1, "NEUTRAL": 2}
        self.is_loaded = False
        
        # ✅ NEW: List of words that usually mean Neutral
        self.neutral_keywords = [
            # Basic Neutral
            "average", "okay", "decent", "fine", "mediocre",

            # Mixed / Contrast Words
            "but", "however", "although", "though", "yet", "on the other hand",

            # Balanced Opinions
            "not bad", "not great", "could be better", "could have been better",
            "good but", "bad but", "nice but", "interesting but",

            # One-time / Forgettable
            "one-time watch", "watchable", "time pass", "forgettable",

            # Emotional Balance
            "mixed", "so-so", "hit or miss", "neither good nor bad",

            # Effort but Weak Impact
            "tries hard", "decent attempt", "fair effort", "not impressive",

            # Story / Direction Related
            "slow in parts", "dragged a bit", "weak ending", "predictable",

            # Positive + Negative Combo Indicators
            "good acting but", "great visuals but", "strong start but",
            "nice concept but", "good first half but",

            # Improvement Hints
            "needed more depth", "needed better editing", 
            "needed stronger writing", "needed tighter pacing"
]



    def load_model(self, adapter_path: str = "./model"):
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Model folder not found at '{adapter_path}'")

        print("=" * 50)
        print("🚀 Loading Sentiment Analysis Model...")
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=3,
            id2label=self.id2label,
            label2id=self.label2id
        )

        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        device = 0 if torch.cuda.is_available() else -1
        
        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        self.is_loaded = True
        print("✅ Model Loaded Successfully!")
        print("=" * 50)

    def _process_prediction(self, result, text):
        """Helper function with Logic + Keyword Override"""
        raw_label = result['label']
        score = result['score'] * 100
        text_lower = text.lower()

        # --- 🔍 DEBUG PRINT ---
        print(f"DEBUG: '{text[:30]}...' -> {raw_label} ({score:.2f}%)")

        final_sentiment = raw_label.upper()

        # 1️⃣ KEYWORD OVERRIDE ( The "Smart Filter" )
        # If the text contains specific neutral words, override the model
        for word in self.neutral_keywords:
            if word in text_lower:
                print(f"   ⚠️ Found neutral keyword '{word}'. Forcing NEUTRAL.")
                return "NEUTRAL", score # Fake high confidence

        # 2️⃣ CONFIDENCE THRESHOLD
        # If model is confused (< 60%), default to Neutral
        if score < 60.0:
            print(f"   ⚠️ Low confidence ({score:.2f}%). Forcing NEUTRAL.")
            final_sentiment = "NEUTRAL"
        
        return final_sentiment, round(score, 2)

    def predict(self, text: str) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        result = self.pipe(text)[0]
        sentiment, confidence = self._process_prediction(result, text)

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        }

    def predict_batch(self, texts: list) -> list:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        results = self.pipe(texts, batch_size=8)
        output = []
        
        for text, result in zip(texts, results):
            sentiment, confidence = self._process_prediction(result, text)

            output.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence
            })

        return output

sentiment_model = SentimentModel()