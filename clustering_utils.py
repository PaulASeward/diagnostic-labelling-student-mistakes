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
        'KMeans': 'KMeans (Default) - K-Means Clustering using Euclidean distance',
        'DBSCAN': 'DBSCAN - Density-Based Spatial Clustering of Applications with Noise',
    }
    return [{'label': value, 'value': key} for key, value in techniques.items()]


def scale_data_to_array(data):
    """
    Scale the given data which should be in the form of a list of lists (or arrays).

    Parameters:
        data (list of lists): Data to be scaled.

    Returns:
        ndarray: Scaled data as a numpy array.
    """
    scaler = StandardScaler()
    try:
        data_array = np.array(data)
        data_scaled = scaler.fit_transform(data_array)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return None

    return data_scaled


def load_input_data(df, col_name, label_column='mistake_category_label'):
    """
    Load and preprocess the input data from a specified column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column containing input data to be processed.
        label_column (str): The name of the label column to be added/checked.

    Returns:
        list: A list of processed data suitable for clustering, or the original DataFrame in case of an error.
        indices: The valid indices of the data in the DataFrame.
    """
    if label_column not in df.columns:
        df[label_column] = pd.NA

    input_data = df[col_name]
    if input_data is None or input_data.empty:
        print(f"Input data is missing or empty from column: {col_name}")
        return df

    # Convert from string representations to lists if necessary
    if isinstance(input_data.iloc[0], str):
        def safe_literal_eval(s):
            try:
                return ast.literal_eval(s)
            except ValueError:
                print(f"Skipping malformed data: {s}")
                return np.nan  # or use a default value or strategy appropriate to your data

        input_data = input_data.apply(safe_literal_eval)

    # Dropping NaN values and preserving the indices of the valid rows
    valid_indices = input_data.dropna().index.tolist()
    input_data = input_data.dropna()

    # Ensure all data is in list or array form
    if not isinstance(input_data.iloc[0], (list, np.ndarray)):
        print(f"Data in column {col_name} is not list or ndarray")
        return df

    return input_data.tolist(), valid_indices


def calculate_centroid(df):
    try:
        data = df['category_hint_embedding'].apply(eval if isinstance(df['category_hint_embedding'].iloc[0], str) else lambda x: x)
        data_ls = data.tolist()
        data_stack = np.stack(data_ls)
        centroid = np.mean(data_stack, axis=0)

        # centroid = np.mean(np.stack(df['category_hint_embedding'].tolist()), axis=0)
        # Find the closest data point to this centroid
        closest_idx = df['category_hint_embedding'].apply(lambda x: np.linalg.norm(np.array(eval(x) if isinstance(x, str) else x) - centroid)).idxmin()
        return df.loc[closest_idx, 'category_hint'], df.loc[closest_idx, 'category_hint_embedding']
    except Exception as e:
        print(f"Error computing centroid or finding closest point: {e}")
        return "Unnamed Cluster", None  # Default name if something goes wrong


class ClusterAlgorithm:
    def __init__(self, clustering_technique, n_clusters=5, **kwargs):
        self.clustering_technique = clustering_technique
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.cluster_algorithm = None
        self.optimal_n_clusters = None
        self.mistake_categories_dict = {}  # Dictionary of {name: embedding}
        self.use_manual_mistake_categories = 0
        self.label_column = 'mistake_category_label'
        self.cluster_name_column = 'mistake_category_name'

    def cluster(self, X):
        input_data, valid_indices = load_input_data(X, col_name='category_hint_embedding', label_column=self.label_column)
        input_data_scaled = scale_data_to_array(input_data)

        if self.use_manual_mistake_categories == 1 and self.mistake_categories_dict: # Use the provided manual category label as a suggestion to initialize Kmeans algorithm.
            category_names = list(self.mistake_categories_dict.keys())
            initial_centers = np.array([np.array(ast.literal_eval(value) if isinstance(value, str) else value) for value in (self.mistake_categories_dict[name] for name in category_names)])

            self.cluster_algorithm = KMeans(n_clusters=len(initial_centers), init=initial_centers, n_init=1, **self.kwargs)
            predicted_categories = self.cluster_algorithm.fit_predict(input_data_scaled)

            X.loc[valid_indices, self.label_column] = predicted_categories
            return X
        elif self.use_manual_mistake_categories == 2 and self.mistake_categories_dict:  # Sort data points to closest provided mistake label
            category_names = list(self.mistake_categories_dict.keys())
            initial_centers = np.array([np.array(ast.literal_eval(value) if isinstance(value, str) else value) for value in (self.mistake_categories_dict[name] for name in category_names)])
            distance_to_centers = np.linalg.norm(input_data_scaled[:, np.newaxis] - initial_centers, axis=2)
            closest_centers_indices = np.argmin(distance_to_centers, axis=1)
            X.loc[valid_indices, self.label_column] = closest_centers_indices

        elif self.clustering_technique == 'KMeans':
            if self.n_clusters == -1:
                sse = []
                for k in range(1, 11):
                    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                    kmeans.fit(input_data_scaled)
                    sse.append(kmeans.inertia_)

                knee_locator = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing')
                self.optimal_n_clusters = knee_locator.knee if knee_locator.knee else 4

            n_clusters = self.optimal_n_clusters if self.optimal_n_clusters else self.n_clusters

            self.cluster_algorithm = KMeans(n_clusters=n_clusters, n_init=10, **self.kwargs)

            predicted_categories = self.cluster_algorithm.fit_predict(input_data_scaled)
            X.loc[valid_indices, self.label_column] = predicted_categories
            return X

        elif self.clustering_technique == 'DBSCAN':
            min_samples = max(2, int(len(X) * 0.05))
            self.cluster_algorithm = DBSCAN(eps=0.5, min_samples=min_samples, **self.kwargs)
            db_labels = self.cluster_algorithm.fit_predict(input_data_scaled)
            X.loc[valid_indices, self.label_column] = np.where(db_labels == -1, X.loc[valid_indices, self.label_column], db_labels)
            return X
        else:
            raise ValueError("Unsupported dimensionality reduction method.")

    def choose_labels(self, X):
        if self.cluster_name_column not in X.columns:
            X[self.cluster_name_column] = np.nan

        if self.use_manual_mistake_categories == 1 and self.mistake_categories_dict:
            category_names = list(self.mistake_categories_dict.keys())
            initial_centers = np.array([np.array(ast.literal_eval(value) if isinstance(value, str) else value) for value in (self.mistake_categories_dict[name] for name in category_names)])

            for i, mistake_category_idx in enumerate(X[self.label_column].unique()):
                # This does not create clusters, but rather instead sorts the points to the closest provided label.
                distances = np.linalg.norm(initial_centers - self.cluster_algorithm.cluster_centers_[mistake_category_idx], axis=1)
                closest_category = list(self.mistake_categories_dict.keys())[np.argmin(distances)]  # Find the closest manual category by embedding distance
                X.loc[X[self.label_column] == mistake_category_idx, self.cluster_name_column] = closest_category
            return X

        if self.use_manual_mistake_categories == 2 and self.mistake_categories_dict:
            category_names = list(self.mistake_categories_dict.keys())
            # Assuming X[self.label_column] contains indices corresponding to category_names
            X[self.cluster_name_column] = X[self.label_column].apply(lambda idx: category_names[idx])
            return X

        self.mistake_categories_dict = {}
        for i, mistake_category_idx in enumerate(X[self.label_column].unique()):
            mistake_category_df = X[X[self.label_column] == mistake_category_idx]

            if not mistake_category_df['category_hint'].mode().empty:  # Find most common mistake category name

                mistake_category_name = mistake_category_df['category_hint'].mode()[0]
                mistake_category_embedding = mistake_category_df[mistake_category_df['category_hint'] == mistake_category_name]['category_hint_embedding'].iloc[0]
            else:
                mistake_category_name, mistake_category_embedding = calculate_centroid(mistake_category_df)

            self.mistake_categories_dict[mistake_category_name] = mistake_category_embedding
            X.loc[X[self.label_column] == mistake_category_idx, self.cluster_name_column] = mistake_category_name

        return X
