# src/train_agent_only.py
"""
Train an empathy classifier using AGENT-REPLY ONLY (normalized text).
Saves:
  - models/empathy_nn_5class.h5
  - artifacts/tfidf_vectorizer.joblib
  - artifacts/label_encoder.joblib
  - artifacts/metadata.json
"""

import os, json, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# ---------- CONFIG (tune for speed / quality) ----------
CSV_PATH = "data/raw/emotions_data.csv"   # change if your CSV has a different name
AGENT_COL_CHOICES = ["labels", "empathetic_dialogues", "response", "agent_reply"]
N_PER_CLASS = 400        # reduce (200) if CPU is slow
MAX_FEATURES = 10000     # reduce (5000) if memory limited
EPOCHS = 6
BATCH = 64
RANDOM_STATE = 42
# -------------------------------------------------------

def normalize_text(s):
    if pd.isna(s): return ""
    s = str(s)
    # Normalize unicode punctuation to ascii-like equivalents
    s = s.replace("’","'").replace("“",'"').replace("”",'"').replace("–","-").replace("\u2013","-")
    s = s.replace("\u2019","'").replace("\xa0"," ")
    # remove extra whitespace
    s = re.sub(r'\s+',' ', s).strip()
    return s.lower()

def label_empathy(text):
    # heuristic labeling function (modify to taste)
    text = normalize_text(text)
    high = ["i'm here","i am here","i'm so sorry","i am so sorry","i'm sorry","i am sorry",
            "i understand","i can imagine","that sounds","must be hard","that must be","i'm with you",
            "i'm here for you","let me help","i will help","if you want to talk","i'm really sorry"]
    supportive = ["glad","congrats","congratulations","appreciate","thanks","happy for you","well done"]
    low = ["ok","fine","sure","alright","noted","noted."]
    very_low = ["whatever","your fault","deal with it","idc","not my problem","so what"]
    for w in high:
        if w in text:
            return "high"
    for w in supportive:
        if w in text:
            return "supportive"
    for w in low:
        if w in text:
            return "low"
    for w in very_low:
        if w in text:
            return "very_low"
    return "neutral"

def find_agent_col(df):
    for c in AGENT_COL_CHOICES:
        if c in df.columns:
            return c
    # fallback: try to guess any text-like column with many characters
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if text_cols:
        # pick column with 'reply' or 'agent' substring if present
        for c in text_cols:
            if "reply" in c.lower() or "agent" in c.lower() or "labels" in c.lower():
                return c
        return text_cols[0]
    raise ValueError("No suitable agent reply column found in CSV. Update AGENT_COL_CHOICES or CSV.")

def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print("Columns found:", df.columns.tolist())

    agent_col = find_agent_col(df)
    print("Using agent reply column:", agent_col)

    # Normalize agent text
    df['agent_text_norm'] = df[agent_col].astype(str).map(normalize_text)

    # Create empathy_level via heuristic labeler
    df['empathy_level'] = df[agent_col].map(label_empathy)

    # Show counts
    counts = df['empathy_level'].value_counts(dropna=False)
    print("Empathy level distribution (raw):\n", counts)

    # Build balanced subset
    levels = ["very_low","low","neutral","supportive","high"]
    parts = []
    for lvl in levels:
        sub = df[df['empathy_level']==lvl]
        if len(sub)==0:
            print(f"Warning: no rows for class {lvl}")
            continue
        take = min(N_PER_CLASS, len(sub))
        parts.append(sub.sample(take, random_state=RANDOM_STATE))
    if not parts:
        raise ValueError("No data selected for training. Check labeling or CSV.")
    df_small = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print("Using balanced subset shape:", df_small.shape)
    print(df_small['empathy_level'].value_counts())

    # Train/test split
    X = df_small['agent_text_norm'].values
    y = df_small['empathy_level'].values
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2,
                                                                random_state=RANDOM_STATE,
                                                                stratify=y)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec  = vectorizer.transform(X_test).toarray()

    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test  = le.transform(y_test_raw)
    y_train_oh = to_categorical(y_train)
    y_test_oh  = to_categorical(y_test)

    # Class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {int(i): float(w) for i, w in zip(classes, class_weights)}
    print("Classes:", list(le.classes_), "Class weights:", cw)

    # Model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_vec.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Fit
    model.fit(X_train_vec, y_train_oh, epochs=EPOCHS, batch_size=BATCH,
              validation_split=0.1, class_weight=cw, verbose=1)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_vec, y_test_oh, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    y_pred = np.argmax(model.predict(X_test_vec, verbose=0), axis=1)
    print("Macro-F1:", f1_score(y_test, y_pred, average='macro'))
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

    # Save artifacts
    model.save("models/empathy_nn_5class.h5")
    joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.joblib")
    joblib.dump(le, "artifacts/label_encoder.joblib")
    with open("artifacts/metadata.json","w") as f:
        json.dump({"classes": list(le.classes_)}, f, indent=2)
    print("Saved model and artifacts.")

if __name__ == "__main__":
    main()
