# 🧠 Sentiment Analysis using NLP

This project is a machine learning application that performs sentiment analysis on textual data using Natural Language Processing (NLP) techniques. It classifies the sentiment of the given text into Positive, Negative, or Neutral. The model is trained using Scikit-learn and traditional NLP techniques like TF-IDF and Naive Bayes.

## 📁 Project Structure

```
sentiment-analysis-nlp/
├── data/
│ └── dataset.csv
├── models/
│ └── sentiment_model.pkl
├── SentimentAnalysis.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Tech Stack

- Python
- NLTK / TextBlob
- Scikit-learn
- Pandas & NumPy
- Matplotlib / Seaborn
- Jupyter Notebook

## ✅ Features

- Text preprocessing: cleaning, tokenization, stopword removal, lemmatization
- Feature extraction using TF-IDF vectorizer
- Sentiment classification using ML algorithms (e.g., Logistic Regression, Naive Bayes)
- Evaluation metrics: Accuracy, Confusion Matrix
- Visualization of results
- Prediction of custom user input

## 🚀 How to Run

1. Clone the repository  
   `git clone https://github.com/Punit231/Sentiment-Analysis.git`

2. Navigate to the project directory  
   `cd Sentiment-Analysis`

3. Install dependencies  
   `pip install -r requirements.txt`

4. Open the notebook  
   `jupyter notebook SentimentAnalysis.ipynb`

5. Run all cells to train and test the model, and try custom sentiment prediction.

## 📈 Sample Output

- Accuracy: 88%
- Confusion Matrix
- Sentiment for input: "I love this product!" → **Positive**

## 💡 Future Scope

- Upgrade to Deep Learning (LSTM or BERT models)
- Web app using Flask or Streamlit
- Real-time sentiment analysis using social media API

---

> Made with ❤️ by Punit Parmar
