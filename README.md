# Sentiment Analysis with BERT on Tweets

This project applies **BERT (Bidirectional Encoder Representations from Transformers)** to perform **sentiment analysis** on a dataset of tweets.  
The dataset contains tweets labeled with sentiment classes, and the goal is to train a transformer-based model to classify the sentiment effectively.  

---

## 📊 Dataset
The dataset consists of:
- **id**: Unique identifier for each tweet
- **label**: Sentiment label (0 = negative, 1 = positive)
- **tweet**: Text of the tweet  

Example:

| id | label | tweet |
|----|-------|-------|
| 1  | 0     | #fingerprint #Pregnancy Test ... |
| 2  | 0     | Finally a transparent silicon case ^^ ... |
| 3  | 0     | We love this! Would you go? #talk #makememories ... |

---

## ⚙️ Steps
1. **Data Preprocessing**
   - Clean tweets (remove links, hashtags, special characters)
   - Tokenize using BERT tokenizer
   - Pad/Truncate sequences

2. **Modeling**
   - Load pretrained `bert-base-uncased`
   - Add classification head
   - Fine-tune on the tweet dataset

3. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix

---

## 📦 Requirements
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install scikit-learn
