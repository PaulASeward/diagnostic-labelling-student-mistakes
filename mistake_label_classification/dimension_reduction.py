from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.preprocessing import StandardScaler


def available_dimension_reduction_techniques():
    """
    Return a list of available dimensionality reduction techniques
    as options for a Dash dropdown component.

    Returns:
    A list of dictionaries, where each dictionary has 'label' and 'value' keys.
    """
    techniques = {
        'PCA': 'PCA (Default) - Principal Component Analysis',
        't-SNE': 't-SNE - t-Distributed Stochastic Neighbor Embedding',
        'MDS': 'MDS - Multi-Dimensional Scaling',
        'Isomap': 'Isomap - Isometric Mapping',
        'KernelPCA': 'KernelPCA - Kernel Principal Component Analysis',
    }
    return [{'label': value, 'value': key} for key, value in techniques.items()]


class ReductionTechnique:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.reducer = None

    def fit(self, X):
        if self.method == 'PCA':
            self.reducer = PCA(**self.kwargs)
        elif self.method == 't-SNE':
            self.reducer = TSNE(**self.kwargs)
        elif self.method == 'MDS':
            self.reducer = MDS(**self.kwargs)
        elif self.method == 'Isomap':
            self.reducer = Isomap(**self.kwargs)
        elif self.method == 'KernelPCA':
            self.reducer = KernelPCA(**self.kwargs, kernel='poly')
        else:
            raise ValueError("Unsupported dimensionality reduction method.")

        self.reducer.fit(X)

    def transform(self, X):
        if self.reducer is None:
            raise ValueError("Call 'fit' before 'transform'.")

        if self.method == 'PCA':
            return self.reducer.transform(X)

        return self.reducer.fit_transform(X)


def reduce_dimensionality(embedding_df, reduction_technique: ReductionTechnique, scaled_data=True):
    if not scaled_data:
        embedding_df = StandardScaler().fit_transform(embedding_df)

    reduction_technique.fit(embedding_df)
    reduced_embedding_df = reduction_technique.transform(embedding_df)

    return reduced_embedding_df


def project_embeddings_to_reduced_dimension(task_df, embedding_array, embedding_type_prefix, technique='PCA', reduced_dimensions=2):
    if embedding_array is not None:
        reduced_embeddings = reduce_dimensionality(embedding_array, reduction_technique=ReductionTechnique(method=technique, n_components=reduced_dimensions))
        for i in range(reduced_dimensions):
            task_df[f'reduced_{embedding_type_prefix}_embedding_{i+1}'] = reduced_embeddings[:, i].tolist()
    return task_df