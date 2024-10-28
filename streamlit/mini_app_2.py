import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Function to load and preprocess data
def load_data():
    url = "Austin_Cultural_Survey_Data.csv"  # Replace with your dataset URL
    data = pd.read_csv(url)
    data.dropna(subset=['Response', 'Auditor-assigned Category'], inplace=True)
    return data

# Function to separate positive and negative responses
def separate_responses(data):
    positive_responses = data[data['Auditor-assigned Category'] == 'Positive']['Response']
    negative_responses = data[data['Auditor-assigned Category'] == 'Negative']['Response']
    return positive_responses.tolist(), negative_responses.tolist()

# Function to cluster responses and extract keywords
def cluster_responses(responses, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(responses)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_, vectorizer.get_feature_names_out(), kmeans.cluster_centers_, X

# Function to extract unique keywords for each cluster
def get_keywords_per_cluster(labels, centers, feature_names, n_keywords=5):
    keywords = {}
    for i in range(centers.shape[0]):
        center = centers[i]
        top_indices = center.argsort()[-n_keywords:][::-1]  # Top keywords
        keywords[i] = [feature_names[index] for index in top_indices]
    return keywords

# Function to plot clusters with keywords in the legend
def plot_clusters_with_keywords(labels, centers, feature_names, keywords):
    plt.figure(figsize=(12, 6))
    unique_labels = np.unique(labels)

    # Calculate the mean position for centering the plot
    cluster_centers = np.array([centers[label] for label in unique_labels if label != -1])
    mean_x = np.mean(cluster_centers[:, 0])

    for label in unique_labels:
        if label == -1:  # Skip noise label if it exists
            continue

        cluster_points = np.where(labels == label)[0]
        # Increase size of the scatter points
        plt.scatter(cluster_points, np.zeros_like(cluster_points) + label, alpha=0.6, s=100, label=f'Cluster {label}')
        
        # Plot cluster center
        plt.scatter(label, np.mean(cluster_points), s=500, color='red', marker='x', label=f'Center of Cluster {label}')
    
    # Set x-axis limits to start at 0
    plt.xlim(0, len(unique_labels) + 1)
    plt.ylim(-0.5, len(unique_labels) - 0.5)

    # Add keywords to the legend
    legend_labels = [f'Cluster {label}: {", ".join(keywords[label])}' for label in unique_labels if label != -1]
    plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.title("Clusters with Unique Keywords")
    plt.xlabel("Clusters")
    plt.yticks([])
    plt.tight_layout()
    st.pyplot()

def main():
    st.title("Response Analysis App")

    data = load_data()
    positive_responses, negative_responses = separate_responses(data)

    st.subheader("Positive Responses")
    st.write(positive_responses)

    st.subheader("Negative Responses")
    st.write(negative_responses)

    n_clusters = st.slider("Select number of clusters", 2, 10, 5)

    if st.button("Cluster Responses"):
        positive_labels, positive_features, positive_centers, positive_X = cluster_responses(positive_responses, n_clusters)
        negative_labels, negative_features, negative_centers, negative_X = cluster_responses(negative_responses, n_clusters)

        # Extract keywords for each cluster
        positive_keywords = get_keywords_per_cluster(positive_labels, positive_centers, positive_features)
        negative_keywords = get_keywords_per_cluster(negative_labels, negative_centers, negative_features)

        # Plot clusters for positive responses
        st.subheader("Positive Responses Clusters")
        plot_clusters_with_keywords(positive_labels, positive_centers, positive_features, positive_keywords)

        # Plot clusters for negative responses
        st.subheader("Negative Responses Clusters")
        plot_clusters_with_keywords(negative_labels, negative_centers, negative_features, negative_keywords)

if __name__ == "__main__":
    main()
