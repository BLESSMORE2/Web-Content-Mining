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
def perform_clustering(data, categories):
    # Vectorize text data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['summary'])

    # Perform clustering
    kmeans = KMeans(n_clusters=len(categories), random_state=42)
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

        # Define categories
        categories = ['Business', 'Politics', 'Arts/Culture/Celebrities', 'Sports', 'Uncategorized']

        # Perform clustering
        clustered_data = perform_clustering(news_df, categories)

        # Display categories
        category_choice = st.sidebar.selectbox("Choose Category", categories)

        filtered_data = clustered_data[clustered_data['cluster'] == categories.index(category_choice)]

        for index, row in filtered_data.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"{row['summary']}")
            st.markdown(f"[Read more]({row['link']})")
    else:
        st.write("Please select at least one news source.")

if __name__ == '__main__':
    main()
