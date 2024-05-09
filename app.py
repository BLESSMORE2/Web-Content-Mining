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
        news_list.append({'title': title, 'link': link, 'summary': summary})
    return pd.DataFrame(news_list)

# Load data for selected news sources
def load_data(sources):
    data = []
    for source in sources:
        data.append(fetch_news(source))
    return pd.concat(data, ignore_index=True)

# Perform clustering
def perform_clustering(data):
    # Vectorize text data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['summary'])

    # Perform clustering
    k = min(5, len(data))  # Number of clusters (limit to the number of articles)
    kmeans = KMeans(n_clusters=k, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)

    return data

# Streamlit App
def main():
    st.title("News Categorization App")
    st.subheader("News Clustered into Categories")

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

        # Perform clustering
        clustered_data = perform_clustering(news_df)

        # Display clusters
        for cluster_id in clustered_data['cluster'].unique():
            st.subheader(f"Cluster {cluster_id + 1}")
            cluster_articles = clustered_data[clustered_data['cluster'] == cluster_id]
            for index, row in cluster_articles.iterrows():
                st.write(f"**{row['title']}**")
                st.write(f"{row['summary']}")
                st.markdown(f"[Read more]({row['link']})")
    else:
        st.write("Please select at least one news source.")

if __name__ == '__main__':
    main()
