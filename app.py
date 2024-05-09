import streamlit as st
import feedparser
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Define category keywords with variations
category_keywords = {
    'Business': ['business', 'economy', 'finance','price','profit','sales', 'market', 'trade', 'stocks', 'company'],
    'Politics': ['politics', 'government', 'war','election', 'policy', 'congress', 'president', 'democracy'],
    'Arts/Culture/Celebrities': ['art', 'culture', 'entertainment', 'celebrity', 'music', 'film', 'artist', 'festival'],
    'Sports': ['sports', 'football', 'soccer', 'basketball', 'tennis', 'athletics','cricket', 'athlete', 'tournament']
}

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
    # Calculate TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(data['summary'])

    # Create a custom TF-IDF matrix with category-based weights
    category_weights = np.zeros(X.shape[1])
    for category, keywords in category_keywords.items():
        category_indices = [vectorizer.vocabulary_.get(keyword.lower(), -1) for keyword in keywords]
        category_indices = [idx for idx in category_indices if idx != -1]
        category_weights[category_indices] = 1

    X_weighted = X.multiply(category_weights)

    # Perform clustering
    kmeans = KMeans(n_clusters=len(category_keywords), random_state=42)
    data['cluster'] = kmeans.fit_predict(X_weighted)

    return data

# Map category choice to cluster containing relevant information
def map_category_to_cluster(category_choice, category_keywords, clusters, vectorizer):
    category_vector = vectorizer.transform([category_keywords[category_choice]])
    similarity_scores = cosine_similarity(clusters, category_vector)
    most_similar_cluster_idx = np.argmax(similarity_scores)
    return most_similar_cluster_idx

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

        # Calculate TF-IDF matrix for categories
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        category_texts = list(category_keywords.values())
        category_vectors = vectorizer.fit_transform([' '.join(words) for words in category_texts])

        # Display categories
        category_choice = st.sidebar.selectbox("Choose Category", list(category_keywords.keys()))

        most_similar_cluster_idx = map_category_to_cluster(category_choice, category_keywords, category_vectors, vectorizer)
        filtered_data = clustered_data[clustered_data['cluster'] == most_similar_cluster_idx]

        for index, row in filtered_data.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"{row['summary']}")
            st.markdown(f"[Read more]({row['link']})")
    else:
        st.write("Please select at least one news source.")

if __name__ == '__main__':
    main()
