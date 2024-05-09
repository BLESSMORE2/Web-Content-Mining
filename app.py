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
        'Business': ['economy', 'business', 'stocks', 'market', 'trade'],
        'Politics': ['politics', 'election', 'senate', 'congress', 'law'],
        'Arts/Culture/Celebrities': ['art', 'movie', 'celebrity', 'theatre', 'culture'],
        'Sports': ['sports', 'game', 'tournament', 'match', 'Olympics']
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
    st.title("News Categorization App")
    st.subheader("News Clustered into Business, Politics, Arts/Culture/Celebrities, and Sports")

    # Select news sources
    selected_sources = st.multiselect("Select News Sources", ['BBC', 'New York Times', 'Guardian', 'Washington Post', 'Times of India'])

    # Fetch news data
    if selected_sources:
        sources_urls = {
            'BBC': 'http://feeds.bbci.co.uk/news/rss.xml',
            'New York Times': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
            'Guardian': 'https://www.theguardian.com/world/rss',
            'Washington Post': 'https://www.washingtonpost.com/rss-national.xml',
            'Times of India': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms'
        }
        news_df = load_data([sources_urls[source] for source in selected_sources])

        # Display categories
        category_choice = st.sidebar.selectbox("Choose Category", ['Politics', 'Business', 'Arts/Culture/Celebrities', 'Sports', 'Uncategorized'])
        filtered_data = news_df[news_df['category'] == category_choice]

        for index, row in filtered_data.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"{row['summary']}")
            st.markdown(f"[Read more]({row['link']})")
    else:
        st.write("Please select at least one news source.")

if __name__ == '__main__':
    main()
