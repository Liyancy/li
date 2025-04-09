"""import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'http://quotes.toscrape.com/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

quotes_data = []
quotes = soup.find_all('div', class_='quote')

for quote in quotes:
    text = quote.find('span', class_='text').get_text()
    author = quote.find('small', class_='author').get_text()
    quotes_data.append({'text': text, 'author': author})

df = pd.DataFrame(quotes_data)
print(df.head())"""
#web scraping
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://news.ycombinator.com/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

titles = soup.select('.titleline')
scores = soup.select('.score')

data = []

for i, title in enumerate(titles):
    post_title = title.get_text()
    post_link = title.find('a')['href'] if title.find('a') else ''
    score = scores[i].get_text() if i < len(scores) else '0 points'
    
    data.append({
        'title': post_title,
        'link': post_link,
        'score': score
    })

df = pd.DataFrame(data)
df.to_csv("hackernews_top_posts.csv", index=False)
print(df.head())
#data preprocessing
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_title'] = df['title'].apply(clean_text)
print("successfully")
#sentinment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

df['sentiment_score'] = df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

def label_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['sentiment_score'].apply(label_sentiment)

print(df[['title', 'score', 'sentiment_score', 'sentiment']].head())
#modelling with LDA, preprocess for LDA
import gensim
from gensim import corpora

# Tokenize titles
texts = [title.split() for title in df['clean_title']]

# Create dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#train LDA
from gensim.models import LdaModel

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, random_state=42, passes=10)

# Display topics
topics = lda_model.print_topics(num_words=5)
for i, topic in topics:
    print(f"Topic {i + 1}: {topic}")
#Assign dominant topic for each post
def get_topic(text):
    bow = dictionary.doc2bow(text.split())
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0] if topics else None

df['topic'] = df['clean_title'].apply(get_topic)
df.to_csv("hackernews_trends.csv", index=False)

#streamlit code for app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load your data

df = pd.read_csv("hackernews_trends.csv")



# --- Sidebar filters ---
st.sidebar.title(" Filters")
topics = df['topic'].unique()
selected_topic = st.sidebar.selectbox("Select Topic", ["All"] + sorted(map(str, topics)))

# Filter by topic
if selected_topic != "All":
    df = df[df['topic'] == int(selected_topic)]

# --- Sentiment Distribution ---
st.title("Stealth Sentiment & Trends Monitor")
st.subheader("Sentiment Distribution")

sentiment_counts = df['sentiment'].value_counts()

fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm", ax=ax)
ax.set_ylabel("Count")
ax.set_xlabel("Sentiment")
st.pyplot(fig)

# --- Word Cloud ---
st.subheader("Word Cloud of Post Titles")
text = " ".join(df['clean_title'].dropna())
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)
st.image(wordcloud.to_array(), use_column_width=True)

# --- Data Table ---
st.subheader("Hacker News Posts")
st.dataframe(df[['title', 'score', 'sentiment', 'topic']].sort_values(by='score', ascending=False))

# --- Footer ---
st.markdown("---")
st.markdown("Built with  using Streamlit and BeautifulSoup")






