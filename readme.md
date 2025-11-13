A neural network trained on TF-IDF text features performs the classification. The project includes a complete pipeline for data preprocessing, feature extraction, model training, evaluation, and command-line interaction.

---

---

## 2. Overview of Components

### 2.1 Artifacts  
Contains all saved model components required for inference:
- **tfidf_vectorizer.joblib**: TF-IDF feature extractor  
- **label_encoder.joblib**: Encodes empathy labels  
- **metadata.json**: Contains class information  

### 2.2 Models  
Contains trained neural network models:
- **empathy_nn_5class.h5**: Saved Keras model for prediction  

### 2.3 Data  
Includes raw training data:
- **emotions_data.csv**: Customer–agent dialogue dataset (place in `data/raw/`)  

### 2.4 Source Code  
Contains training scripts, CLI applications, and notebooks:
- **train_agent_only.py**  
  Trains the model using only the agent's reply (normalized text).  
- **train_quick.py**  
  Fast training script for rapid experimentation.  
- **cli_empathy.py**  
  Basic command-line predictor.  
- **cli_empathy_fixed.py**  
  Improved command-line system with text normalization and rule-based overrides.  
- **Empathy_Level_Prediction.ipynb**  
  Interactive notebook demonstrating pipeline and experimentation.  

---

## 3. Empathy Classification Framework

The system classifies an agent’s reply into one of the following five categories:

1. **High**  
   Demonstrates strong emotional understanding and explicit empathetic support.
2. **Supportive**  
   Shows encouragement, validation, or appreciation.
3. **Neutral**  
   Informational or factual responses without emotional engagement.
4. **Low**  
   Brief or cold responses lacking emotional acknowledgment.
5. **Very Low**  
   Rude, hostile, dismissive, or harmful responses.

The classification is based solely on the **agent’s reply**, making the system robust even when the customer message is minimal or unclear.

---

## 4. Installation and Setup

### 4.1 Prerequisites
- Python 3.10 or later  
- Conda or Virtual Environment (recommended)  
- Windows, macOS, or Linux  


---

## 5. Training the Model

Two training options are available.

### 5.1 Full Training (Recommended)

This script:
- Normalizes agent responses  
- Generates heuristic empathy labels  
- Builds TF-IDF vectors  
- Trains a neural network  
- Saves model and preprocessing artifacts  

### 5.2 Quick Training (Small Dataset)

Useful for low-resource systems or debugging.

After training, confirm the following files are created:
- `models/empathy_nn_5class.h5`  
- `artifacts/tfidf_vectorizer.joblib`  
- `artifacts/label_encoder.joblib`

---

## 6. Running the Command Line Interface

### 6.1 Basic CLI

### 6.2 Enhanced CLI (recommended)


This improved version includes:
- Normalization of punctuation  
- Detection of empathetic and dismissive keywords  
- Rule-based override for strong empathetic expressions  
- Detailed probability and cue output  

### 6.3 Interactive Usage Example
Once running, the CLI will prompt:


Output includes:
- Predicted empathy level  
- Tone of the agent reply  
- Explanation/note  
- Probability distribution  
- Detected cues  

---

## 7. Evaluating the Model

Place test dialogues into a CSV file and use an evaluation script such as:


This will:
- Generate predictions  
- Compare against expected labels  
- Produce a confusion matrix and classification report  

---

## 8. How the Model Works

1. **Text Normalization**  
   Special characters and punctuation are standardized.

2. **TF-IDF Vectorization**  
   Converts text into numerical feature vectors.

3. **Label Encoding**  
   Empathy labels are converted into numerical classes.

4. **Neural Network Architecture**  
   - Dense layer with ReLU activation  
   - Dropout for regularization  
   - Output softmax layer for five empathy classes  

5. **Prediction**  
   Softmax probabilities are computed for each class, and the class with the highest probability is chosen.

6. **Rule-Based Enhancement (cli_empathy_fixed.py)**  
   Certain explicit empathetic phrases override model uncertainty for improved interpretability.

---

## 9. Limitations

- Keyword-based heuristic labeling may introduce noise.  
- TF-IDF models struggle with sarcasm, context, and subtle tone.  
- The model evaluates only the agent’s reply, not conversation context.  
- Deep contextual models (BERT, DistilBERT) could further improve accuracy.

---

## 10. Future Enhancements

- Incorporating transformer-based embeddings (BERT/DistilBERT).  
- Multi-turn context-aware empathy detection.  
- Integration into a real-time chatbot interface.  
- Dataset expansion with manually corrected labels.  
- Emotion + Empathy joint classification system.

---

## 11. Conclusion

This project demonstrates the application of natural language processing and neural networks to classify the empathy level in agent responses. It provides both a functional command-line tool and a research-oriented pipeline suitable for academic evaluation. The modular design enables extensibility for future improvements in model performance and interpretability.

---

If you require a full project report or a slide presentation, these can be prepared from the materials and results produced by this repository.
"""

def main():
    target = "README.md"
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(README_CONTENT)
        print(f"README.md created successfully at ./{target}")
    except Exception as e:
        print("Failed to write README.md:", e)

if __name__ == "__main__":
    main()