import os
import streamlit as st
import feedparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Fetch and parse the RSS feed for selected news source
def fetch_news(source_url):
    feed = feedparser.parse(source_url)
    news_list = []
    for entry in feed.entries:
        title = entry.get('title', 'No Title Available')
        link = entry.get('link', 'No URL Available')
        summary = entry.get('summary', entry.get('description', 'No Summary Available'))
        category = classify_news(title, summary)
        news_list.append({'title': title, 'link': link, 'summary': summary, 'category': category})
    return pd.DataFrame(news_list)

# Classify news based on keywords
def classify_news(title, summary):
    keywords = {
        'Business':  ['business', 'economy', 'finance', 'price','loss', 'profit', 'sales', 'market', 'trade', 'stocks', 'company','shares'],
        'Politics': ['politics', 'election','senate', 'congress', 'law','government', 'war', 'election', 'policy', 'congress', 'president', 'democracy'],
        'Arts/Culture/Celebrities': ['art', 'movie', 'celebrity', 'theatre', 'culture','art', 'culture', 'entertainment', 'celebrity', 'music', 'film', 'artist', 'festival'],
        'Sports': ['sports', 'game', 'match', 'Olympics', 'football', 'soccer', 'basketball', 'tennis', 'athletics', 'cricket', 'athlete', 'tournament']
    }
    text = title.lower() + ' ' + summary.lower()
    for category, words in keywords.items():
        if any(word in text for word in words):
            return category
    return 'Uncategorized'

# Load data for selected news sources
def load_data(sources):
    data = []
    for source in sources:
        data.append(fetch_news(source))
    return pd.concat(data, ignore_index=True)

# Streamlit App
def main():
    st.title("News Categorization and Clustering App")
    st.subheader("News Clustered into Business, Politics, Arts/Culture/Celebrities, and Sports")

    # Select news sources
    selected_sources = st.multiselect("Select News Sources", ['BBC', 'New York Times', 'CNN', 'Washington Post', 'Times of India'])

    # Fetch news data
    if selected_sources:
        sources_urls = {
            'BBC': 'http://feeds.bbci.co.uk/news/rss.xml',
            'New York Times': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
            'CNN': 'http://rss.cnn.com/rss/cnn_topstories.rss',
            'Washington Post': 'https://www.washingtonpost.com/rss-national.xml',
            'Times of India': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms'
        }
        news_df = load_data([sources_urls[source] for source in selected_sources])

        # Display categories
        st.sidebar.subheader("Choose Category")
        category_choice = st.sidebar.selectbox("Select Category", ['Business', 'Politics', 'Arts/Culture/Celebrities', 'Sports', 'Uncategorized'])
        filtered_data = news_df[news_df['category'] == category_choice]

        for index, row in filtered_data.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"{row['summary']}")
            st.markdown(f"[Read more]({row['link']})")

        # Store data in CSV
        csv_filename = "news_data.csv"
        csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
        news_df.to_csv(csv_path, index=False)
        st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        st.sidebar.markdown(f"**Save data:**")
        st.sidebar.markdown(f"Download the CSV file [here]({csv_filename})")
    else:
        st.write("Please select at least one news source.")

if __name__ == '__main__':
    main()
