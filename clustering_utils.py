import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from kneed import KneeLocator
import ast
from sklearn.preprocessing import StandardScaler


def cluster_student_mistakes_kmeans(df, embedding_type_prefix='feedback'):
    label_column = f'mistake_category_index_from_{embedding_type_prefix}_embedding'
    if label_column not in df.columns:
        df.loc[:, label_column] = pd.NA

        # Safely accessing and converting input data
    input_data = df.get(f'{embedding_type_prefix}_embedding')
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

    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(input_data_scaled)
        sse.append(kmeans.inertia_)

    knee_locator = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing')
    optimal_k = knee_locator.knee if knee_locator.knee else 3  # Default to 3 if no knee is found

    # Running KMeans with the optimal number of clusters
    optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    predicted_categories = optimal_kmeans.fit_predict(input_data_scaled)

    df[label_column] = predicted_categories

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

    min_samples = max(2, int(len(df) * min_samples_ratio))
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    predicted_categories = dbscan.fit_predict(input_data_scaled)

    df[label_column] = predicted_categories

    return df
