import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff
import plotly.express as px

COLOR_PALETTE = px.colors.qualitative.Plotly
CATEGORY_NAME_COL = 'mistake_category_name'
CATEGORY_IDX_COL = 'mistake_category_label'
TEXT_COL = "category_hint"


def create_color_map(task_embeddings_df, mistake_categories_dict, color_palette=COLOR_PALETTE):
    sorted_indices = sorted(task_embeddings_df[CATEGORY_IDX_COL].unique())
    color_map = {idx: color_palette[idx % len(color_palette)] for idx in sorted_indices}
    return color_map


# def create_color_map(task_embeddings_df, mistake_categories_dict, color_palette=COLOR_PALETTE):
#     name_to_idx = {row[CATEGORY_NAME_COL]: row[CATEGORY_IDX_COL] for index, row in task_embeddings_df.iterrows()}
#     sorted_names = sorted(mistake_categories_dict.keys())
#     return {name_to_idx[name]: color_palette[i % len(color_palette)] for i, name in enumerate(sorted_names)}


def build_scatter_plot_with_mistake_category_trace(task_embeddings_df, mistake_categories_dict, jitter=0.02):
    title = f'Visualizing Clusters of Mistake Labels from Reduced Embeddings'
    color_map = create_color_map(task_embeddings_df, mistake_categories_dict)

    fig = go.Figure()
    for _, mistake_category_idx in enumerate(task_embeddings_df[CATEGORY_IDX_COL].unique()):
        mistake_category_df = task_embeddings_df[task_embeddings_df[CATEGORY_IDX_COL] == mistake_category_idx]

        marker_color = color_map[mistake_category_idx]  # Get color for Mistake Category Type
        custom_data = mistake_category_df.apply(lambda row: {'student_id': row['student_id'], 'category_hint_idx': row['category_hint_idx']}, axis=1).tolist()

        x_values = mistake_category_df[f'reduced_category_hint_embedding_1'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_category_hint_embedding_1'].max() - task_embeddings_df[f'reduced_category_hint_embedding_1'].min())
        y_values = mistake_category_df[f'reduced_category_hint_embedding_2'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_category_hint_embedding_2'].max() - task_embeddings_df[f'reduced_category_hint_embedding_2'].min())

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers', marker=dict(color=marker_color, line=dict(width=1, color='DarkSlateGrey')),
            name=mistake_category_df[CATEGORY_NAME_COL].iloc[0], text=mistake_category_df[TEXT_COL], customdata=custom_data, hoverinfo='text+name'
        ))

    fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', clickmode='event+select', width=1200, height=800,
                      title={'text': title,'y': 0.9,'x': 0.5,'xanchor': 'center','yanchor': 'top','font': {'size': 20, 'color': 'black', 'family': "Arial"}},
                      legend=dict(title=dict(text='Mistake Category', side='top')))
    return fig


def plot_mistake_statistics(df, mistake_categories_dict):
    """
    Creates a pie chart and a bar chart figure showing the distribution of student mistakes by category.

    Parameters:
        df (pd.DataFrame): DataFrame containing the mistake categories and their counts.
        mistake_categories_dict (Dictionary): name: embedding

    Returns:
        figure: The pie chart figure.
    """
    if CATEGORY_NAME_COL not in df.columns or CATEGORY_IDX_COL not in df.columns:
        raise ValueError(f"The DataFrame must contain the columns '{CATEGORY_NAME_COL}' and '{CATEGORY_IDX_COL}'.")

        # Count the occurrences of each category
    df_count = df.groupby([CATEGORY_NAME_COL, CATEGORY_IDX_COL]).size().reset_index(name='count')

    color_map = create_color_map(df, mistake_categories_dict)
    df_count['color'] = df_count[CATEGORY_IDX_COL].apply(lambda idx: color_map[idx])
    color_discrete_map = {row[CATEGORY_NAME_COL]: row['color'] for index, row in df_count.iterrows()}

    # Generate Pie Chart
    pie_fig = px.pie(df_count, names=CATEGORY_NAME_COL, values='count', title='Distribution of Student Mistakes (Pie Chart)', color_discrete_map=color_discrete_map)
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(width=800, height=800)

    return pie_fig


def plot_dendrogram(task_embeddings_df, mistake_categories_dict):
    """
    Plots a dendrogram using Plotly to show hierarchical clustering of embeddings to visualize similarity between student mistakes.

    Parameters:
        task_embeddings_df (pd.DataFrame): DataFrame containing the embeddings and labels.
        mistake_categories_dict (Dictionary): name: embedding
    """
    embeddings_columns = ['reduced_category_hint_embedding_1', 'reduced_category_hint_embedding_2']
    color_map = create_color_map(task_embeddings_df, mistake_categories_dict)
    embeddings = task_embeddings_df[embeddings_columns].values
    #label_names = task_embeddings_df[CATEGORY_NAME_COL].values
    labels = task_embeddings_df[CATEGORY_IDX_COL].values

    # Compute the linkage matrix using Ward's method
    linkage_matrix = linkage(embeddings, 'ward')

    # Create the dendrogram with colors
    dendro = ff.create_dendrogram(linkage_matrix, orientation='left')
    # for i, d in enumerate(dendro['data']):
    #     # Update colors based on the labels of the dendrogram
    #     dy = d['y']
    #     df = task_embeddings_df.iloc[dy]
    #     category_idx = df.mistake_category_label.values[0]
    #     d['line']['color'] = color_map[category_idx]

    fig = go.Figure(data=dendro['data'])
    fig.update_layout(title_text='Dendrogram of Student Mistakes with Color Coding',
                      xaxis=dict(title='Euclidean Distance'),
                      yaxis=dict(title='Student Mistakes'),
                      width=800, height=800)

    return fig

