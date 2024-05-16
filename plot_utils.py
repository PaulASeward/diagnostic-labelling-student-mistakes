import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff
import plotly.express as px

COLOR_PALETTE = px.colors.qualitative.Plotly
CATEGORY_NAME_COL = 'mistake_category_name'
CATEGORY_IDX_COL = 'mistake_category_label'
TEXT_COL = "category_hint"


# dendro = ff.create_dendrogram(embeddings, orientation='left', labels=label_names, colorscale=list(color_map.values()), linkagefun=lambda x: linkage(x, 'ward', optimal_ordering=True))


def create_color_map(task_embeddings_df, mistake_categories_dict, color_palette=COLOR_PALETTE):
    sorted_indices = sorted(task_embeddings_df[CATEGORY_IDX_COL].unique())
    color_map = {}
    for idx in sorted_indices:     # Map both index and category name to the same color
        color = color_palette[idx % len(color_palette)]
        category_name = task_embeddings_df[task_embeddings_df[CATEGORY_IDX_COL] == idx][CATEGORY_NAME_COL].iloc[0]
        # category_name = mistake_categories_dict.get(idx, "Unknown")
        color_map[idx] = color
        color_map[category_name] = color

    print("Color Map", color_map)
    return color_map


def build_scatter_plot_with_mistake_category_trace(task_embeddings_df, mistake_categories_dict, color_map, jitter=0.02):
    """
    Creates a scatter plot showing the groupings of student mistakes by their reduced embeddings.

    Parameters:
        df (pd.DataFrame): DataFrame containing the mistake categories and their counts.
        mistake_categories_dict (Dictionary): name: embedding
        color_map (Dictionary): Access colour through name or index of mistake category
        jitter (float): Randomizes the X & Y coordinates to prevent overlap.
    Returns:
        figure: The scatter plot figure.
    """
    title = f'Visualizing Clusters of Mistake Labels from Reduced Embeddings'

    fig = go.Figure()
    for mistake_category_idx in sorted(task_embeddings_df[CATEGORY_IDX_COL].unique()):
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

    fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', clickmode='event+select', width=1200, height=750,
                      title={'text': title,'y': 0.9,'x': 0.5,'xanchor': 'center','yanchor': 'top','font': {'size': 20, 'color': 'black', 'family': "Arial"}},
                      legend=dict(title=dict(text='Mistake Category', side='top')))
    return fig


def plot_mistake_statistics(task_embeddings_df, mistake_categories_dict, color_map):
    """
    Creates a pie chart and a bar chart figure showing the distribution of student mistakes by category.

    Parameters:
        task_embeddings_df (pd.DataFrame): DataFrame containing the mistake categories and their counts.
        mistake_categories_dict (Dictionary): name: embedding
        color_map (Dictionary): Access colour through name or index of mistake category
    Returns:
        figure: The pie chart figure.
    """
    df_count = task_embeddings_df.groupby([CATEGORY_NAME_COL]).size().reset_index(name='count')

    # Generate Pie Chart
    pie_fig = px.pie(df_count, names=CATEGORY_NAME_COL, values='count', title='Stemblytics Student Mistake Breakdown', color=CATEGORY_NAME_COL, color_discrete_map=color_map, hole=.2)
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(width=750, height=750, showlegend=False)

    return pie_fig


def plot_dendrogram(task_embeddings_df, mistake_categories_dict, color_map):
    """
    Plots a dendrogram using Plotly to show hierarchical clustering of embeddings to visualize similarity between student mistakes.

    Parameters:
        task_embeddings_df (pd.DataFrame): DataFrame containing the embeddings and labels.
        mistake_categories_dict (Dictionary): name: embedding
        color_map (Dictionary): Access colour through name or index of mistake category
    """
    embeddings_columns = ['reduced_category_hint_embedding_1', 'reduced_category_hint_embedding_2']
    embeddings = task_embeddings_df[embeddings_columns].values

    label_names = task_embeddings_df[CATEGORY_NAME_COL].values
    label_idx = task_embeddings_df[CATEGORY_IDX_COL].values

    # Create the dendrogram with colors
    dendro = ff.create_dendrogram(embeddings, orientation='left', labels=label_names)

    fig = go.Figure(data=dendro['data'])
    fig.update_layout(title_text='Student Mistake Categories Dendrogram',
                      xaxis=dict(title='Distance'),
                      # yaxis=dict(title='Student Mistake Label'),
                      yaxis=dict(title='Student Mistake Label', ticktext=label_names),
                      width=1200, height=1200, margin=dict(l=200))

    return fig

