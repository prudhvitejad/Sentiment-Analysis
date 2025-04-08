# Sentiment-Analysis

## Algorithm

```c
1. Input: .csv file // raw dataset  
2. Output: sentiment-polarity  
3. PROCEDURE Sentiment_Analysis  
4.     Dataset ← Load_CSV(path_to_csv_file)  
5.     Text_Cleaning(Dataset) // Preprocess text by removing URLs, special characters, and converting to lowercase  
6.     Labels_Encoding(Dataset) // Convert sentiment labels to numerical values  

    // Tokenization using RoBERTa  
7.     Tokens ← Tokenize(Dataset, max_length=128, padding=True, truncation=True)  

    // Extracting Features using Pretrained RoBERTa Model  
8.     RoBERTa_Model ← Load_Pretrained_RoBERTa()  
9.     FOR EACH tokenized_text IN Tokens DO:  
10.         Embeddings ← RoBERTa_Model(tokenized_text) // Extract contextual embeddings  
11.     END FOR  

    // Define Deep Learning Model with LSTM, BiLSTM, GRU, and MultiHeadAttention  
12.     RobertaEnsembleModel ← Initialize()  
13.     LSTM_output ← Apply_LSTM(Embeddings, units=128, return_sequences=True)  
14.     BiLSTM_output ← Apply_Bidirectional_LSTM(Embeddings, units=128, return_sequences=True)  
15.     GRU_output ← Apply_GRU(Embeddings, units=128, return_sequences=True)  

    // Apply MultiHeadAttention for each recurrent layer  
16.     LSTM_attention ← MultiHeadAttention(LSTM_output, LSTM_output)  
17.     BiLSTM_attention ← MultiHeadAttention(BiLSTM_output, BiLSTM_output)  
18.     GRU_attention ← MultiHeadAttention(GRU_output, GRU_output)  

    // Merge Features from different Attention Mechanisms  
19.     Merged_Features ← Concatenate(LSTM_attention, BiLSTM_attention, GRU_attention)  

    // Fully Connected Layer  
20.     Dropout_Layer ← Apply_Dropout(Merged_Features, rate=0.3)  
21.     Output ← Dense(Dropout_Layer, activation="Softmax", units=3)  

    // Sentiment Prediction  
24.     FOR EACH Test_Sample DO:  
25.         Sentiment_Score ← Model_Predict(Test_Sample)  
26.         Sentiment_Class ← Argmax(Sentiment_Score)  
27.         IF Sentiment_Score ≥ Threshold THEN:  
28.             Sentiment_Polarity ← Classify(Sentiment_Class) // Negative, Neutral, Positive  
29.         END IF  
30.     END FOR  
31. END PROCEDURE
```

## 🧩 Pipeline Overview

- **Input:** IMDb Movie Reviews Dataset (CSV)
- **Preprocessing:**
  - Text Cleaning (URLs, mentions, punctuations, etc.)
  - Tokenization using RoBERTa tokenizer
  - Padding & Truncation to a fixed length
  - Label Encoding (Sentiment → Numeric)
- **Model:** Ensembled Model (RoBERTa + LSTM + BiLSTM + GRU + Multi-Head Attention)
- **Output:** Predicted Sentiment Class




## 🛠️ Modules Used

| Module             | Purpose                                       |
|--------------------|-----------------------------------------------|
| `re`               | Text cleaning and preprocessing               |
| `pandas`           | Data loading and manipulation                 |
| `numpy`            | Numerical operations                          |
| `transformers`     | RoBERTa tokenizer and pretrained model        |
| `tensorflow`       | Deep learning model definition and training   |
| `sklearn`          | Train-test split and label encoding           |


## 🧠 Algorithms Used

### ✅ RoBERTa: Contextual Embedding
- Transformer-based model pretrained on a large corpus.
- Generates contextualized vector embeddings for each word/token.

### 🔁 LSTM / BiLSTM / GRU: Sequence Modeling
- LSTM: Learns time-step dependencies in a forward direction.
- BiLSTM: Learns both forward and backward dependencies.
- GRU: Lightweight variant for efficient sequence modeling.

### ✨ Multi-Head Attention: Feature Refinement
- Allows the model to focus on different semantic aspects of the sequence.
- Enhances feature interactions and context awareness.

### 🧮 Dense Layer (Softmax): Classification
- Final output layer with `softmax` activation to predict sentiment class probabilities.
