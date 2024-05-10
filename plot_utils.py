import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
import plotly.express as px


def get_color_for_category(index, color_palette=px.colors.qualitative.Plotly):
    return color_palette[index % len(color_palette)]


def add_traces_by_mistake_category(fig, task_embeddings_df, color_design, embedding_type_prefix, jitter):
    mistake_label_column = "mistake_category_label"
    for i, mistake_category in enumerate(task_embeddings_df[mistake_label_column].unique()):
        mistake_category_df = task_embeddings_df[task_embeddings_df[mistake_label_column] == mistake_category]
        marker_color = get_color_for_category(i, color_design)  # Get color for TA
        text_column = "category_hint"

        x_values = mistake_category_df[f'reduced_{embedding_type_prefix}_embedding_1'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_1'].max() - task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_1'].min())
        y_values = mistake_category_df[f'reduced_{embedding_type_prefix}_embedding_2'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_2'].max() - task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_2'].min())

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers', marker=dict(color=marker_color, line=dict(width=1, color='DarkSlateGrey')),
            name=str(mistake_category), text=mistake_category_df[text_column], hoverinfo='text+name'
        ))
    return fig


def build_scatter_plot_with_mistake_category_trace(task_embeddings_df, embedding_type_prefix,  jitter=0.01):
    title = f'Visualizing Clusters of Mistake Labels from {embedding_type_prefix} Embeddings'

    fig = go.Figure()
    color_palette = px.colors.qualitative.Plotly

    fig = add_traces_by_mistake_category(fig, task_embeddings_df, color_palette, embedding_type_prefix, jitter)

    fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', clickmode='event+select', width=1600, height=800,
                      title={'text': title,'y': 0.9,'x': 0.5,'xanchor': 'center','yanchor': 'top','font': {'size': 20, 'color': 'black', 'family': "Arial"}},
                      legend=dict(# x=1.1,# y=0.5,# xanchor='right',# yanchor='middle',# orientation='v',
                          title=dict(text='Mistake Category', side='top')),)
    return fig