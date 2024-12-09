import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns





# Load the Yelp dataset
@st.cache_data 
def load_data(): 
    return pd.read_json('data/yelp_dataset/truncated_yelp_academic_dataset_review.json')

data = load_data()

def compute_regression_metrics():
    data['review_length'] = data['text'].apply(lambda x: len(x.split()))
    X = data[['review_length']]  # Now the independent variable is review length
    y = data['stars']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return [mse, mae, r2]

# Function to calculate Dunn Index
def dunn_index(distances, labels):
    unique_labels = set(labels)
    inter_cluster_distances = []
    intra_cluster_distances = []

    for label in unique_labels:
        cluster_points = [distances[i] for i in range(len(labels)) if labels[i] == label]
        if len(cluster_points) < 2:
            continue
        intra_cluster_distance = pdist(cluster_points)
        intra_cluster_distances.extend(intra_cluster_distance)

    for label1 in unique_labels:
        for label2 in unique_labels:
            if label1 < label2:  # Avoid duplicate comparisons
                cluster1 = [distances[i] for i in range(len(labels)) if labels[i] == label1]
                cluster2 = [distances[j] for j in range(len(labels)) if labels[j] == label2]
                if cluster1 and cluster2:
                    inter_cluster_distances.append(min(pdist(cluster1 + cluster2)))

    if not inter_cluster_distances or not intra_cluster_distances:
        return None

    return min(inter_cluster_distances) / max(intra_cluster_distances)

def compute_clustering_metrics(n_clusters):
    sample_reviews = data['text'].sample(100, random_state=42)  # Sample of 100 reviews

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sample_reviews)

   
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
        
        # Get labels and calculate metrics
    labels = kmeans.labels_
        
        # Silhouette Score
    silhouette = silhouette_score(X, labels)
        
        # Adjusted Rand Index (requires ground truth labels, e.g., stars)
        # Here we're using stars as ground truth labels for illustration
    y_true = data['stars'].sample(100, random_state=42)  # Sample corresponding star ratings
    rand_index = adjusted_rand_score(y_true, labels)
        
        # Dunn Index
    distances = squareform(pdist(X.toarray()))  # Convert sparse matrix to dense for distance calculation
    dunn = dunn_index(distances, labels)

    return [silhouette, dunn, rand_index]
    
def compute_classification_metrics():
    sample_reviews = data['text'].sample(500, random_state=42)  # Sample of 100 reviews
    sample_labels = data['stars'].sample(500, random_state=42)  # Sample corresponding star ratings (e.g., binary classification)

    # Assuming a binary classification based on star ratings (1 to 3: Negative, 4 to 5: Positive)
    binary_labels = (sample_labels > 3).astype(int)  # Convert to binary (0 or 1)
    model = LogisticRegression()
    vectorizer = TfidfVectorizer(stop_words='english')
    model.fit(vectorizer.fit_transform(sample_reviews), binary_labels)
    predictions = model.predict(vectorizer.transform(sample_reviews))

        # Calculate metrics
    accuracy = accuracy_score(binary_labels, predictions)
    precision = precision_score(binary_labels, predictions)
    recall = recall_score(binary_labels, predictions)
    auc_roc = roc_auc_score(binary_labels, predictions)

    return [accuracy, precision, recall, auc_roc]

def mean_average_precision(y_true, y_scores):
    """Calculate Mean Average Precision (MAP)"""
    # Sort by scores
    sorted_indices = np.argsort(-y_scores)
    sorted_y_true = y_true[sorted_indices]

    # Calculate Average Precision
    ap = 0.0
    num_hits = 0
    for i, p in enumerate(sorted_y_true):
        if p == 1:
            num_hits += 1
            ap += num_hits / (i + 1)
    return ap / max(num_hits, 1)

def ndcg(y_true, y_scores, k):
    """Calculate Normalized Discounted Cumulative Gain (NDCG)"""
    # Sort by scores
    sorted_indices = np.argsort(-y_scores)
    sorted_y_true = y_true[sorted_indices][:k]

    # Calculate DCG
    dcg = sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(sorted_y_true))
    
    # Ideal DCG
    ideal_sorted_y_true = np.sort(y_true)[::-1][:k]
    ideal_dcg = sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_sorted_y_true))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def precision_at_k(y_true, k):
    """Calculate Precision at K"""
    return np.mean(y_true[:k])

def compute_ranking_metrics(k):
    sample_reviews = data['text'].sample(100, random_state=42)  # Sample of 100 reviews
    relevance_scores = (data['stars'].sample(100, random_state=42) > 3).astype(int)  # Relevance based on star ratings
    scores = np.random.rand(len(relevance_scores))  # Random scores for demonstration

        # Calculate metrics
    map_score = mean_average_precision(relevance_scores.values, scores)
    ndcg_score = ndcg(relevance_scores.values, scores, k)  # Example k=10
    precision_k = precision_at_k(relevance_scores.values, k)  # Example k=10

    return [map_score, ndcg_score, precision_k]

def landing_page():
    st.subheader("Welcome to the AI Metrics Dashboard")
    st.write("""
        This dashboard provides various metrics to evaluate AI models based on different use cases:
        
        - **Data Assessment**: Understand the shape and structure of unprocessed, raw data. 
        - **Classification**: Evaluate model performance using accuracy, precision, recall, and AUC-ROC.
        - **Regression**: Analyze regression model metrics like Mean Absolute Error (MAE) and R-squared.
        - **Clustering**: Assess clustering performance with silhouette score, Rand index, and Dunn index.
        - **Ranking**: Measure ranking performance using Mean Average Precision (MAP), NDCG, and Precision at K.
        - **Dynamic Assessment**: Track shifts in data in real time, and measure model performance over time across evaluation strategies.
        
        Use the sidebar to navigate to different sections and explore the metrics. The demos of individual metrics acts on data pulled from [The Yelp Dataset](https://www.yelp.com/dataset), but all of these metrics can apply to your own AI models via the [Reward Reports](https://reward-reports-a0f858c3c007.herokuapp.com/) documentation interface! Our dynamic assessment feature is trained on real city data drawn from the Austin, Texas and Boston, Massachussets. 
    """)


def plot_star_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='stars', data=data, palette='viridis')
    plt.title("Distribution of Star Ratings")
    plt.xlabel("Star Ratings")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    st.pyplot(plt)

def plot_review_length_distribution(data):
    # Calculate the length of each review
    data['review_length'] = data['text'].apply(len)

    plt.figure(figsize=(10, 6))
    sns.histplot(data['review_length'], bins=30, kde=True, color='skyblue')
    plt.title("Distribution of Review Lengths")
    plt.xlabel("Length of Review (Characters)")
    plt.ylabel("Frequency")
    st.pyplot(plt)
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to load and preprocess data
def load_real_data():
    url = "Austin_Cultural_Survey_Data.csv"  # Replace with your dataset URL
    data = pd.read_csv(url)
    data.dropna(subset=['Response', 'Auditor-assigned Category'], inplace=True)
    return data

# Function to separate positive and negative responses
def separate_responses(data):
    positive_responses = data[data['Auditor-assigned Category'] == 'Positive']
    negative_responses = data[data['Auditor-assigned Category'] == 'Negative']
    return positive_responses, negative_responses

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
# Function to plot clusters with keywords in the legend
def plot_clusters_with_keywords(labels, centers, feature_names, keywords):
    fig, ax = plt.subplots(figsize=(12, 6))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:  # Skip noise label if it exists
            continue

        cluster_points = np.where(labels == label)[0]
        # Increase size of the scatter points
        ax.scatter(cluster_points, np.zeros_like(cluster_points) + label, alpha=0.6, s=100, label=f'Cluster {label}')

        # Plot a translucent halo around the cluster
        ax.scatter(cluster_points, np.zeros_like(cluster_points) + label, 
                   alpha=0.2, s=400, color=plt.cm.tab10(label), edgecolor='w')

        # Plot cluster center
        ax.scatter(label, np.mean(cluster_points), s=500, color='red', marker='x', label=f'Center of Cluster {label}')
    
    # Set x-axis limits to start at 0
    ax.set_xlim(0, len(unique_labels) + 1)
    ax.set_ylim(-0.5, len(unique_labels) - 0.5)

    # Add keywords to the legend
    legend_labels = [f'Cluster {label}: {", ".join(keywords[label])}' for label in unique_labels if label != -1]
    ax.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.set_title("Clusters with Unique Keywords")
    ax.set_xlabel("Clusters")
    ax.set_yticks([])
    
    plt.tight_layout()
    st.pyplot(fig)


# Function to train a classifier and evaluate metrics
def evaluate_classifier(data, response_col, category_col, increment):
    precision_scores = []
    accuracy_scores = []
    increments = []
    
    # Prepare the data for the model
    X = data[response_col].values
    y = data[category_col].map({'Positive': 1, 'Negative': 0}).values  # Map categories to binary

    for i in range(increment, len(X) + 1, increment):
        X_subset = X[:i]
        y_subset = y[:i]
        
        vectorizer = TfidfVectorizer(stop_words='english')
        X_vec = vectorizer.fit_transform(X_subset)

        model = LogisticRegression()
        model.fit(X_vec, y_subset)
        y_pred = model.predict(X_vec)

        precision = precision_score(y_subset, y_pred)
        accuracy = accuracy_score(y_subset, y_pred)

        precision_scores.append(precision)
        accuracy_scores.append(accuracy)
        increments.append(i)

    return increments, precision_scores, accuracy_scores

def prepare_response_counts(data):
    positive_count = data[data['Auditor-assigned Category'] == 'Positive'].shape[0]
    negative_count = data[data['Auditor-assigned Category'] == 'Negative'].shape[0]
    
    return positive_count, negative_count

def evaluate_separate_regressions(data, response_col, category_col, increment):
    positive_counts = []
    negative_counts = []
    increments = []

    for i in range(increment, len(data) + 1, increment):
        subset = data[:i]
        positive_count, negative_count = prepare_response_counts(subset)
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)
        increments.append(i)

    # Reshape the increments for sklearn
    X = np.array(increments).reshape(-1, 1)

    # Fit the regression model for positive counts
    model_positive = LinearRegression()
    model_positive.fit(X, positive_counts)

    # Fit the regression model for negative counts
    model_negative = LinearRegression()
    model_negative.fit(X, negative_counts)
    
    return increments, positive_counts, negative_counts, model_positive, model_negative
# Streamlit app title
st.title("Welcome to the Hortus Trellis")

st.sidebar.title("Hortus Trellis")
st.sidebar.subheader("Select Use Case")
use_case = st.sidebar.radio("Choose a metric type or visualization:", ("Home", "Data", "Classification", "Regression", "Clustering", "Ranking", "Dynamic Assessment"))

if use_case == "Classification":
    st.subheader("Classification Metrics")
    st.write("Classification is valuable when you need to categorize data into predefined labels or classes. In practice, this can look like medical diagnoses, sentiment analysis, blocking spam emails, or detecting pedestrians on a road via computer vision. Some of the most valuable metrics in this space are:")
    st.write("- Accuracy: reflects the ratio of accurately classified data points within the dataset. Straightforward and especially useful when you are expecting the category sizes to be relatively balanced.")
    st.write("- Precision: measures how many positive classifications are true positives. This metric is valuable when the cost of a false positive is high, such as in non-life-threatening medical diagnoses to avoid stress and financial burden.")
    st.write("- Recall: conversely, measures how many negative data points the model accurately classified. This metric matters when false negatives pose more risk, such as in life-threatening diagnostic processes.")
    st.write("- ROC-AUC: summarizes a model's performance across all classification thresholds. Especially valuable for unbalanced datasets, or to compare multiple models using a decision-making threshold.")

    st.write("Let's test out what some of these values might look like if we were to classify Yelp reviews as positive or negative based on the number of stars:")

    if st.button("Calculate Accuracy"):
        st.success(f"Accuracy: {compute_classification_metrics()[0]:.2f}")
    if st.button("Calculate Precision"):
        st.success(f"Precision: {compute_classification_metrics()[1]:.2f}")
    if st.button("Calculate Recall"):
        st.success(f"Recall: {compute_classification_metrics()[2]:.2f}")
    if st.button("Calculate ROC-AUC"):
        st.success(f"ROC-AUC: {compute_classification_metrics()[3]:.2f}")
    


elif use_case == "Regression":
    st.subheader("Regression Metrics")
    st.write("Regression is useful when you would like to predict a numerical value; the model attempts to isolate patterns in the data, and predict what reality would look like for a previously unseen input. Housing prices, future stock prices, cyber risk, weather patterns, and projected sales are all prime examples of regression models in use. Some common metrics include: ")
    st.write("- Mean Squared Error (MSE): an estimate the average distance between the model's prediction and reality. The lower the MSE, the more accurate the predictions are in a regression. MSE is best suited for when outliers are especially harmful, as it is more sensitive to extremes.")
    st.write("- Mean Absolute Error (MAE): another estimate of the magnitude of error between the model prediction and reality. While MSE is sensitive to outliers, MAE is more robust and is easier to interpret on the same scale as the data itself.")
    st.write("- R-squared: indicates how well the algorithm has been able to predict the general trend of the data. The higher the R-squared value, the more closely the model is performing to reality across the range of all data.")

    st.write("Here is an example of what those values would look like if we were trying to predict the number of stars in a Yelp review based off of how long the review itself was:")

    if st.button("Calculate MSE"):
        st.success(f"Mean Squared Error: {compute_regression_metrics()[0]:.2f}")
    if st.button("Calculate MAE"):
        st.success(f"Mean Absolute Error: {compute_regression_metrics()[1]:.2f}")
    if st.button("Calculate R-squared"):
        st.success(f"R-squared: {compute_regression_metrics()[2]:.2f}")
    


elif use_case == "Clustering":
    st.subheader("Clustering Metrics")
    st.write("Clustering is most helpful when you are trying to categorize data without predefined labels. Most likely, you have access to a hige repository of information, and would like to gain insight on the kinds of groups the data itself is comprised of. Standard uses cases include anomaly detection, market segmentation, and distillation. Some example metrics are: ")
    st.write("- Silhouette Score: checks how cohesive the groups themselves are. Values closer to -1 indicate that the clusters are probably misaligned, while +1 means the model has captured the data into well-defined, distinct groups. This metric is optimal for when you expect large, clear groups, as it is sensitive to noise.")
    st.write("- Dunn Index: like the Silhouette Score, but able to parse between smaller, less distinct clusters more succesfully. The downside is that it is more computationally expensive, and can take more processing power to maintain.")
    st.write("- Rand Index: offers a numerical assessment of how much overlap there is between clusters. 1 stands for a perfect overlap, while 0 means completely separate groups. Our implementation takes random luck into account, meaning that if a Rand Index is negative, that means the clustering is less accurate than random assignment.")

    st.write("Another critical component of clustering is the amount of clusters themselves--feel free to toggle between 2 and 10 to see how these metrics shift as you optimize for broader or narrower groups. This is an example of what those values could look like while clustering a Yelp review dataset: ")
    n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3)

    if st.button("Calculate Silhouette Score"):
        st.success(f"Silhouette Score: {compute_clustering_metrics(n_clusters)[0]:.2f}")
    if st.button("Calculate Dunn Index"):
        st.success(f"Dunn Index: {compute_clustering_metrics(n_clusters)[1]:.2f}")
    if st.button("Calculate Rand Index"):
        st.success(f"Rand Index: {compute_clustering_metrics(n_clusters)[2]:.2f}")

elif use_case == "Ranking":
    st.subheader("Ranking Metrics")
    st.write("Ranking metrics are most useful when the order of recommendations generated by a system assess its performance. The practical applications of this are multifold: search engine results, procurement recommendations, case management, and emergency services, to name a few. Some example metrics and their meanings are: ")
    st.write("- Mean Average Precision (MAP): derived from the average of the accuracies of all rankings where all relevant information or touchpoints are included. A higher MAP score indicates that, more often than not, all relevant solutions are being included.")
    st.write("- Normalized Discounted Cumulative Gain (NDCG): compares generated rankings to an ideal where the goal output is at the top of the list. A measure of ranking quality that is best used when you have predefined data on the best possible machine responses.")
    st.write("- Precision at K: checks how many of the relevant solutions are returned within the first k results. This is often applied in situations like search, where users rarely move past the first page. Ideal for relevancy and retention.")

    st.write("Let's try using Yelp review stars as a measure of relevancy, now, in ranked lists. Play around with how relevancy-based metrics change as the number of recommended items grows and drops. Here is what those metrics for assessment would look like:")
    
    k = st.number_input("Number of Recommended Items", min_value=2, max_value=10, value=3)
    if st.button("Calculate MAP"):
        st.success(f"MAP: {compute_ranking_metrics(k)[0]:.2f}")
    if st.button("Calculate NDCG"):
        st.success(f"NDCG: {compute_clustering_metrics(k)[1]:.2f}")
    if st.button("Calculate Precision at K"):
        st.success(f"Precision at K: {compute_clustering_metrics(k)[2]:.2f}")

elif use_case == "Home": 
    landing_page()

elif use_case == "Data":
        try:
            st.subheader("Dataset Overview")
            st.write("The data the metrics currently describe is extracted from [The Yelp Dataset](https://www.yelp.com/dataset). The information that we use to calculate the metrics are primarily the stars, review text, review id, and date. Here is a preview of the information it contains:")
            st.dataframe(data.head(20)) 

            st.write("### Star Ratings Distribution")
            plot_star_distribution(data)

            # Plotting the review length distribution
            st.write("### Review Length Distribution")
            plot_review_length_distribution(data)

        except Exception as e:
            st.error(f"An error occurred: {e}")
elif use_case == "Dynamic Assessment":
    data = load_real_data()
    positive_responses, negative_responses = separate_responses(data)
    combined_data = pd.concat([positive_responses, negative_responses], ignore_index=True)
    st.subheader("Dynamic Assessment of Models and Performance")
    st.write("Here, we have combined real citizen data from the City of Boston and the City of Austin about the availability and function of city services. The data is comprised of binary inputs like thumbs up and down buttons, along with more detailed, nuanced text.")
    st.write("Below, we have sorted the citizen feedback into positive and negative categories, so that we can isolate successful and nonsuccessful avenues for growth with government initiatives:")
    st.write("Example Positive Citizen Feedback:")
    st.dataframe(positive_responses, use_container_width=True)

    st.write("Example Negative Citizen Feedback:")
    st.dataframe(negative_responses, use_container_width=True)


    st.subheader("Clustering Citizen Feedback", divider=True)
    st.write("One critical dimension of analyzing user data is harnessing it to become more proactive about governance. Our clustering functionality allows us to track growing trends and topics in feedback so that cities can respond to public concerns and track effective services early.")
    st.write("In general, clustering is helpful to draw insight from uncategorized, messy datasets. The size of the clusters you select allows you to explore broader trends (fewer clusters) or isolate nuanced details (more clusters). Try toggling with the number of clusters below to see some key words and categories in the citizen feedback:")
    n_clusters = st.slider("Select number of clusters", 2, 10, 5)


    # Existing clustering and plotting code...
    if st.button("Cluster Responses"):
        positive_labels, positive_features, positive_centers, positive_X = cluster_responses(positive_responses['Response'].tolist(), n_clusters)
        negative_labels, negative_features, negative_centers, negative_X = cluster_responses(negative_responses['Response'].tolist(), n_clusters)

        # Extract keywords for each cluster
        positive_keywords = get_keywords_per_cluster(positive_labels, positive_centers, positive_features)
        negative_keywords = get_keywords_per_cluster(negative_labels, negative_centers, negative_features)

        # Plot clusters for positive and negative responses
        st.subheader("Positive Responses Clusters")
        st.write("These are the categories and keywords that are linked most closely to positive feedback:")
        plot_clusters_with_keywords(positive_labels, positive_centers, positive_features, positive_keywords)

        st.write("The legend of the visualization shows the key words that drive each emerging topic. ")

        st.subheader("Negative Responses Clusters")
        st.write("These are the categories and keywords that are linked most closely to negative feedback:")
        plot_clusters_with_keywords(negative_labels, negative_centers, negative_features, negative_keywords)

    st.subheader("Assessing Model Performance Through Regression:", divider=True)
    st.write("We also can use the Hortus Trellis to ensure a model is operating at its best. Here, we see how predict how accurately the model can anticipate good and bad changes in the system with the addition of data, simulating model performance over time. As more data is added, we get a general sense of where the model's success plateaus.")
    st.write("Like the clustering example, you can play around with how the addition of more data provides more information about how well the model is performing:")
    increment = st.slider("Select Data Increment for Response Prediction:", min_value=1, max_value=len(combined_data), value=5, step=1)
    # Regression evaluation section
    if st.button("Evaluate Regression Model"):
        increments, positive_counts, negative_counts, model_positive, model_negative = evaluate_separate_regressions(
            combined_data, 'Response', 'Auditor-assigned Category', increment)

        # Plotting the positive and negative counts
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(increments, positive_counts, label='Predicted True Positives', marker='o', color='green')
        ax.plot(increments, negative_counts, label='Predicted False Positives', marker='o', color='red')
        
        # Predict future values
        future_increments = np.arange(increments[-1] + 1, increments[-1] + 21, 1).reshape(-1, 1)
        positive_future = model_positive.predict(future_increments)
        negative_future = model_negative.predict(future_increments)

        # Plot future predictions
        ax.plot(future_increments, positive_future, linestyle='--', color='lightgreen', label='True Positive Trend')
        ax.plot(future_increments, negative_future, linestyle='--', color='lightcoral', label='False Positive Trend')

        ax.set_title("True and False Positives Over Incremental Data")
        ax.set_xlabel("Number of Responses Used for Training")
        ax.set_ylabel("Count of Responses")
        ax.legend()
        ax.grid()

        st.pyplot(fig)
        st.write("By adding more data to the model training process and creating a regression, we can track how many true positives and false positives the model detects over time. This gives us a sense of how model performance is limited by data as opposed to core functionality, allowing clients to compare models in their projected best state. ")






