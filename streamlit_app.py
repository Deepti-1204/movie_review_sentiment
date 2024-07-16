import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import time
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap

# Ensure stopwords are downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load data
@st.cache_data(persist=True)
def load_data():
    df = pd.read_csv('IMDB Dataset.csv')
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# Preprocess data
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['review'] = df['review'].apply(clean_text)

# Sidebar title
st.sidebar.title('Sentiment Analysis Dashboard')

# Main title
st.title('Interactive Sentiment Analysis Dashboard')

# Show dataset stats
st.subheader('Dataset Overview')
st.write(df.head())

# Visualize sentiment distribution
st.subheader('Sentiment Distribution')
sentiment_counts = df['sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# Word cloud for positive and negative reviews
positive_reviews = " ".join(review for review in df[df['sentiment'] == 1]['review'])
negative_reviews = " ".join(review for review in df[df['sentiment'] == 0]['review'])

st.subheader('Word Clouds')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
wordcloud_positive = WordCloud(max_words=500, background_color="white", colormap='viridis').generate(positive_reviews)
wordcloud_negative = WordCloud(max_words=500, background_color="black", colormap='plasma').generate(negative_reviews)

ax1.imshow(wordcloud_positive, interpolation='bilinear')
ax1.set_title('Positive Reviews')
ax1.axis("off")

ax2.imshow(wordcloud_negative, interpolation='bilinear')
ax2.set_title('Negative Reviews')
ax2.axis("off")

st.pyplot(fig)

# Model training and evaluation
st.subheader('Model Training and Evaluation')

# Sample the data due to memory constraints
df_sample = df.sample(n=5000, random_state=42)
X = df_sample['review']
y = df_sample['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SVM model
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

# Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

# Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

# Model metrics
metrics = {
    'Model': ['SVM', 'Logistic Regression', 'Naive Bayes'],
    'Accuracy': [accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_nb)],
    'F1 Score': [f1_score(y_test, y_pred_svm), f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_nb)],
    'Precision': [precision_score(y_test, y_pred_svm), precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_nb)],
    'Recall': [recall_score(y_test, y_pred_svm), recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_nb)]
}

metrics_df = pd.DataFrame(metrics)

st.write('## Model Evaluation Metrics')
st.table(metrics_df)

# Fine-grained sentiment analysis (example using Vader)
analyzer = SentimentIntensityAnalyzer()
df['vader_score'] = df['review'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['fine_grained_sentiment'] = pd.cut(df['vader_score'], bins=5, labels=['highly negative', 'negative', 'neutral', 'positive', 'highly positive'])

# Visualize fine-grained sentiment distribution
st.subheader('Fine-Grained Sentiment Distribution')
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='fine_grained_sentiment', hue='fine_grained_sentiment', data=df, palette='coolwarm', ax=ax, dodge=False)
ax.set_title('Distribution of Fine-Grained Sentiment Labels')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
st.pyplot(fig)

# SHAP for model interpretation (example using Logistic Regression)
st.subheader('SHAP Model Interpretation')

explainer = shap.Explainer(lr, X_train_tfidf)
shap_values = explainer(X_test_tfidf)

st.set_option('deprecation.showPyplotGlobalUse', False)
shap.summary_plot(shap_values, X_test_tfidf, plot_type='bar')
st.pyplot(bbox_inches='tight')

# Cache for model predictions
prediction_cache = {}

# User input for sentiment prediction
st.subheader('Sentiment Prediction for User Reviews')

user_review = st.text_area('Enter your movie review here:')
if st.button('Predict Sentiment'):
    start_time = time.time()
    with st.spinner('Predicting sentiment...'):
        cleaned_review = clean_text(user_review)

        if cleaned_review in prediction_cache:
            svm_pred, lr_pred, nb_pred = prediction_cache[cleaned_review]
        else:
            tfidf_review = vectorizer.transform([cleaned_review])

            svm_pred = svm.predict(tfidf_review)
            lr_pred = lr.predict(tfidf_review)
            nb_pred = nb_model.predict(tfidf_review)

            prediction_cache[cleaned_review] = (svm_pred, lr_pred, nb_pred)

    end_time = time.time()
    elapsed_time = end_time - start_time

    st.write(f"Review: {user_review}")
    st.write(f"SVM Prediction: {'Positive' if svm_pred[0] == 1 else 'Negative'}")
    st.write(f"Logistic Regression Prediction: {'Positive' if lr_pred[0] == 1 else 'Negative'}")
    st.write(f"Naive Bayes Prediction: {'Positive' if nb_pred[0] == 1 else 'Negative'}")
    st.write(f"Time taken for prediction: {elapsed_time:.2f} seconds")

# Show footer or additional information
st.sidebar.markdown('---')
st.sidebar.markdown('Created by Deepti')
