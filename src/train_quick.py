# src/train_quick.py
import os, json, itertools
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

# --------- CONFIG ----------
CSV_PATH = "data/raw/emotions_data.csv"   # adjust if different
N_PER_CLASS = 600    # lower -> faster; adjust if CPU slow
MAX_FEATURES = 10000 # smaller for faster training
EPOCHS = 6
BATCH = 64
# --------------------------

def label_empathy(text):
    text = str(text).lower()
    high = ["sorry", "understand", "feel", "i'm here", "i’m here", "must be hard", "that sounds tough", "i can imagine", "help"]
    supportive = ["glad", "appreciate", "congrats", "thanks", "happy"]
    low = ["ok", "fine", "sure", "alright"]
    very_low = ["whatever", "your fault", "deal with it", "idc"]
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

def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    # try to find columns
    for c in ["labels", "empathetic_dialogues", "dialogue", "response"]:
        if c in df.columns:
            label_col = c
            break
    else:
        raise Exception("Cannot find agent reply column. Ensure CSV has 'labels' or 'empathetic_dialogues' column.")
    # find emotion or situation column optionally
    text_col = None
    for c in ["empathetic_dialogues","dialogue","agent_reply","labels"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise Exception("Cannot find text column.")

    # Build input_text: if 'emotion' column exists include it, else just use the agent reply
    if "emotion" in df.columns:
        df['input_text'] = df['emotion'].astype(str) + " | " + df[text_col].astype(str)
    else:
        df['input_text'] = df[text_col].astype(str)

    # Create empathy_level via heuristic
    df['empathy_level'] = df[label_col].apply(label_empathy)
    # sample balanced subset
    parts = []
    levels = ["very_low","low","neutral","supportive","high"]
    for lvl in levels:
        sub = df[df['empathy_level']==lvl]
        if len(sub)==0:
            continue
        take = min(N_PER_CLASS, len(sub))
        parts.append(sub.sample(take, random_state=42))
    df_small = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    print("Using dataset shape:", df_small.shape)
    print(df_small['empathy_level'].value_counts())

    X = df_small['input_text'].values
    y = df_small['empathy_level'].values

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec  = vectorizer.transform(X_test).toarray()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test  = le.transform(y_test_raw)
    y_train_oh = to_categorical(y_train)
    y_test_oh  = to_categorical(y_test)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {int(i): float(w) for i, w in zip(classes, class_weights)}
    print("Classes:", list(le.classes_))

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_vec.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_vec, y_train_oh, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, class_weight=cw, verbose=1)

    # evaluate
    y_pred = np.argmax(model.predict(X_test_vec, verbose=0), axis=1)
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))
    # save artifacts
    model.save("models/empathy_nn_5class.h5")
    joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.joblib")
    joblib.dump(le, "artifacts/label_encoder.joblib")
    with open("artifacts/metadata.json","w") as f:
        json.dump({"classes": list(le.classes_)}, f, indent=2)
    print("Saved: models/empathy_nn_5class.h5 and artifacts/*.joblib")

if __name__=="__main__":
    main()
