# src/cli_empathy_fixed.py
import os, re, joblib, numpy as np
from tensorflow.keras.models import load_model

# Paths
VEC_PATH = "artifacts/tfidf_vectorizer.joblib"
LE_PATH  = "artifacts/label_encoder.joblib"
MDL_PATH = "models/empathy_nn_5class.h5"

# Tone map
TONE_MAP = {
    "very_low": ("Dismissive / Hostile", "The reply is dismissive and potentially harmful."),
    "low":      ("Cold / Brief",        "The reply is short and does not acknowledge feelings."),
    "neutral":  ("Neutral / Informational", "The reply is factual but not emotionally supportive."),
    "supportive":("Supportive / Encouraging","The reply gives positive reinforcement or congratulations."),
    "high":     ("Highly Empathetic / Compassionate", "The reply shows strong understanding, validation and offer of support.")
}

EMPATHETIC_PHRASES = [
    "i'm here", "i am here", "i'm so sorry", "i am so sorry", "i'm sorry", "i am sorry",
    "i understand", "i can imagine", "that sounds", "must be hard", "that must be", "i'm with you",
    "i'm here for you", "i will help", "let me help", "if you want to talk"
]

EMPATHETIC = set("sorry understand feel here for you must be hard imagine that sounds tough help concern support".split())
DISMISSIVE = set("ok fine whatever idc deal with it your fault stop calm down not my problem".split())

def normalize(s):
    if s is None: return ""
    s = s.replace("’","'").replace("“",'"').replace("”",'"').replace("–","-").replace("\u2013","-")
    s = s.replace("\u2019","'").replace("\xa0"," ")
    return re.sub(r'\s+',' ', s).strip().lower()

def highlight_words(text):
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    highs = [t for t in tokens if t in EMPATHETIC]
    lows  = [t for t in tokens if t in DISMISSIVE]
    return list(dict.fromkeys(highs))[:6], list(dict.fromkeys(lows))[:6]

def load_artifacts():
    missing = [p for p in (VEC_PATH, LE_PATH, MDL_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))
    vec = joblib.load(VEC_PATH)
    le  = joblib.load(LE_PATH)
    mdl = load_model(MDL_PATH)
    return vec, le, mdl

def pretty_print_probs(class_names, probs):
    pairs = sorted(zip(class_names, probs), key=lambda x: -x[1])
    for name, p in pairs:
        print(f"  {name:<12} : {p:.4f}")

def strong_emp_override(text_norm):
    for ph in EMPATHETIC_PHRASES:
        if ph in text_norm:
            return True
    return False

def main():
    print("Loading model artifacts...")
    try:
        vectorizer, label_enc, model = load_artifacts()
    except Exception as e:
        print("ERROR loading artifacts:", e)
        return
    classes = list(label_enc.classes_)
    print("Loaded. Classes:", classes)
    print("Type 'exit' or 'quit' at any prompt to stop.\n")

    try:
        while True:
            cust = input("Customer message: ").strip()
            if cust.lower() in ("exit","quit"):
                break
            agent = input("Agent reply   : ").strip()
            if agent.lower() in ("exit","quit"):
                break
            if not agent:
                print("Agent reply empty — please type something.\n")
                continue

            # Use agent reply only (normalized)
            agent_norm = normalize(agent)

            # Keyword override: if agent reply contains a strong empathetic phrase, set HIGH immediately
            override_high = strong_emp_override(agent_norm)

            # Predict
            X = vectorizer.transform([agent_norm]).toarray()   # agent-only input
            probs = model.predict(X, verbose=0)[0]
            idx = int(np.argmax(probs))
            pred_label = label_enc.inverse_transform([idx])[0]

            if override_high and pred_label != "high":
                # override prediction to high (soft override)
                pred_label = "high"
                # optionally adjust probs to reflect override
                probs = np.array([0.0]*len(probs))
                # place small prob mass to others but highest to high
                high_index = list(label_enc.classes_).index("high")
                probs[high_index] = 0.9
                probs = probs / probs.sum()

            tone_name, tone_note = TONE_MAP.get(pred_label, ("Unknown",""))

            print("\n---- Prediction ----")
            print(f"Predicted empathy level : {pred_label.upper()}")
            print(f"Tone (human label)      : {tone_name}")
            print(f"Note                    : {tone_note}")
            print("\nProbabilities:")
            pretty_print_probs(list(label_enc.classes_), probs)
            highs, lows = highlight_words(agent)
            print("\nDetected cues:")
            print("  Empathetic cues :", ", ".join(highs) if highs else "None")
            print("  Dismissive cues :", ", ".join(lows) if lows else "None")
            print("---------------------\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")

if __name__ == "__main__":
    main()
