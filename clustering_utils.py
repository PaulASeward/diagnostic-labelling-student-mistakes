import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from kneed import KneeLocator
import ast
from sklearn.preprocessing import StandardScaler


def available_clustering_techniques():
    """
    Return a list of available dimensionality reduction techniques
    as options for a Dash dropdown component.

    Returns:
    A list of dictionaries, where each dictionary has 'label' and 'value' keys.
    """
    techniques = {
        'KMeans': 'KMeans - K-Means Clustering using Euclidean distance',
        'DBSCAN': 'DBSCAN - Density-Based Spatial Clustering of Applications with Noise',
    }
    return [{'label': value, 'value': key} for key, value in techniques.items()]



def cluster_student_mistakes_kmeans(df):
    label_column = f'mistake_category_label'
    if label_column not in df.columns:
        df.loc[:, label_column] = pd.NA

        # Safely accessing and converting input data
    input_data = df.get('category_hint_embedding')
    if input_data is None or input_data.empty:
        print("Input data is missing or empty.")
        return df

    # Convert from string to lists if necessary
    if isinstance(input_data.iloc[0], str):
        try:
            input_data = input_data.apply(ast.literal_eval).tolist()
        except ValueError as e:
            print(f"Error converting input data from string: {e}")
            return df
    else:
        input_data = input_data.tolist()

    scaler = StandardScaler()

    try:
        input_data_scaled = scaler.fit_transform(input_data)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return df  # Return early if scaling fails

    # sse = []
    # for k in range(1, 11):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(input_data_scaled)
    #     sse.append(kmeans.inertia_)
    #
    # knee_locator = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing')
    # optimal_k = knee_locator.knee if knee_locator.knee else 3  # Default to 3 if no knee is found

    # Running KMeans with the optimal number of clusters
    optimal_kmeans = KMeans(n_clusters=5, random_state=42)
    predicted_categories = optimal_kmeans.fit_predict(input_data_scaled)

    df[label_column] = predicted_categories

    return df

def expand_and_cluster_student_mistakes_kmeans(df, embedding_type_prefix='category_hint'):
    embedding_columns = [f'{embedding_type_prefix}_{i}_embedding' for i in range(1, 4)]
    label_column = 'mistake_cluster_labels'
    if label_column not in df.columns:
        df.loc[:, label_column] = pd.NA

    # Expand and Scale the DataFrame so each embedding is treated as a separate data point
    expanded_df = expand_df(df, embedding_columns)
    expanded_array_scaled = scale_data_to_array(expanded_df)

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust clusters as needed
    cluster_labels = kmeans.fit_predict(expanded_array_scaled)

    # sse = []
    # for k in range(1, 11):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(input_data_scaled)
    #     sse.append(kmeans.inertia_)
    #
    # knee_locator = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing')
    # optimal_k = knee_locator.knee if knee_locator.knee else 3  # Default to 3 if no knee is found
    #
    # # Running KMeans with the optimal number of clusters
    # optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    # predicted_categories = optimal_kmeans.fit_predict(input_data_scaled)

    # Map cluster labels back to original DataFrame
    try:
        df = map_cluster_labels_to_df(df, cluster_labels, embedding_columns)
    except Exception as e:
        print(f"Error mapping cluster labels back to DataFrame: {e}")
        return df
    return


def expand_df(df, embedding_columns):
    # Expand the DataFrame so each embedding is treated as a separate data point
    expanded_data = []
    for idx, row in df.iterrows():
        for col in embedding_columns:
            embedding = row[col]
            if isinstance(embedding, str):
                try:
                    embedding = ast.literal_eval(embedding)
                except ValueError:
                    continue  # Skip if the data cannot be converted
            if isinstance(embedding, list):
                expanded_data.append(embedding)

    # Check if expanded data is non-empty
    if not expanded_data:
        print("No valid embeddings available for clustering.")
        return df

    return expanded_data


def scale_data_to_array(df):
    # Convert list to numpy array and scale
    df_array = np.array(df)
    scaler = StandardScaler()
    try:
        df_array_scaled = scaler.fit_transform(df_array)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return df

    return df_array_scaled


def map_cluster_labels_to_df(df, cluster_labels, embedding_columns):
    # Map cluster labels back to original DataFrame
    label_index = 0
    for idx, row in df.iterrows():
        labels = []
        for col in embedding_columns:
            embedding = row[col]
            if isinstance(embedding, str):
                embedding = ast.literal_eval(embedding)
            if isinstance(embedding, list):
                labels.append(cluster_labels[label_index])
                label_index += 1
        df.at[idx, 'mistake_cluster_labels'] = str(labels)

    return df


def cluster_student_mistakes_dbscan(df, embedding_type_prefix='feedback', epsilon=0.5, min_samples_ratio=0.05):
    label_column = f'mistake_category_index_from_{embedding_type_prefix}_embedding'
    if label_column not in df.columns:
        df.loc[:, label_column] = pd.NA

    input_data = df[f'{embedding_type_prefix}_embedding']
    if isinstance(input_data[0], str):
        input_data = input_data.apply(ast.literal_eval).tolist()
    else:
        input_data = input_data.tolist()

    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Check if dynamic min_samples is appropriate ratio
    min_samples = max(2, int(len(df) * min_samples_ratio))
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    predicted_categories = dbscan.fit_predict(input_data_scaled)

    df[label_column] = predicted_categories

    return df
