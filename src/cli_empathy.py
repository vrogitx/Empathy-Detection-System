# src/cli_empathy.py
import os
import re
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# --- Paths: adjust if needed ---
VEC_PATH = "artifacts/tfidf_vectorizer.joblib"
LE_PATH  = "artifacts/label_encoder.joblib"
MDL_PATH = "models/empathy_nn_5class.h5"

# --- Tone map for human-friendly output ---
TONE_MAP = {
    "very_low": ("Dismissive / Hostile", "The reply is dismissive and potentially harmful."),
    "low":      ("Cold / Brief",        "The reply is short and does not acknowledge feelings."),
    "neutral":  ("Neutral / Informational", "The reply is factual but not emotionally supportive."),
    "supportive":("Supportive / Encouraging","The reply gives positive reinforcement or congratulations."),
    "high":     ("Highly Empathetic / Compassionate", "The reply shows strong understanding, validation and offer of support.")
}

# --- Keyword sets for quick highlights ---
EMPATHETIC = set("""
sorry understand feel here for you must be hard imagine that sounds tough help concern support
""".strip().split())
DISMISSIVE = set("""
ok fine whatever idc deal with it your fault stop calm down not my problem
""".strip().split())

def highlight_words(text):
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    highs = [t for t in tokens if t in EMPATHETIC]
    lows  = [t for t in tokens if t in DISMISSIVE]
    # remove duplicates keeping order
    highs = list(dict.fromkeys(highs))
    lows  = list(dict.fromkeys(lows))
    return highs[:6], lows[:6]

def load_artifacts(vec_path=VEC_PATH, le_path=LE_PATH, mdl_path=MDL_PATH):
    missing = []
    if not os.path.exists(vec_path): missing.append(vec_path)
    if not os.path.exists(le_path):  missing.append(le_path)
    if not os.path.exists(mdl_path): missing.append(mdl_path)
    if missing:
        raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))
    vec = joblib.load(vec_path)
    le  = joblib.load(le_path)
    mdl = load_model(mdl_path)
    return vec, le, mdl

def pretty_print_probs(class_names, probs):
    # sorted by descending prob
    pairs = sorted(zip(class_names, probs), key=lambda x: -x[1])
    for name, p in pairs:
        print(f"  {name:<12} : {p:.4f}")

def main():
    print("Loading model artifacts...")
    try:
        vectorizer, label_enc, model = load_artifacts()
    except Exception as e:
        print("ERROR loading artifacts:", e)
        print("Make sure the following files exist (relative to project root):")
        print("  ", VEC_PATH)
        print("  ", LE_PATH)
        print("  ", MDL_PATH)
        return

    class_names = list(label_enc.classes_)
    print("Loaded artifacts. Classes:", class_names)
    print("Type 'exit' or 'quit' at any prompt to stop.\n")

    try:
        while True:
            cust = input("Customer message: ").strip()
            if cust.lower() in ("exit","quit"):
                print("Exiting.")
                break
            agent = input("Agent reply   : ").strip()
            if agent.lower() in ("exit","quit"):
                print("Exiting.")
                break

            if not agent:
                print("Agent reply is empty. Please type something.")
                continue

            # Combined input (context-aware)
            combined = agent.strip()

            # Vectorize & predict
            try:
                X = vectorizer.transform([combined]).toarray()
                probs = model.predict(X, verbose=0)[0]
            except Exception as e:
                print("Prediction error:", e)
                continue

            idx = int(np.argmax(probs))
            pred_label = label_enc.inverse_transform([idx])[0]
            tone_name, tone_note = TONE_MAP.get(pred_label, ("Unknown", ""))

            print("\n---- Prediction ----")
            print(f"Predicted empathy level : {pred_label.upper()}")
            print(f"Tone (human label)      : {tone_name}")
            print(f"Note                    : {tone_note}")
            print("\nProbabilities:")
            pretty_print_probs(class_names, probs)
            highs, lows = highlight_words(agent)
            print("\nDetected cues:")
            print("  Empathetic cues :", ", ".join(highs) if highs else "None")
            print("  Dismissive cues :", ", ".join(lows) if lows else "None")
            print("---------------------\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    except Exception as e:
        print("Unexpected error:", e)

if __name__ == "__main__":
    main()
