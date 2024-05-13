import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
import plotly.express as px

COLOR_PALETTE = px.colors.qualitative.Plotly


def get_color_for_category(mistake_category_idx, color_palette=COLOR_PALETTE):
    return color_palette[mistake_category_idx % len(color_palette)]


def add_traces_by_mistake_category(fig, task_embeddings_df, embedding_type_prefix, jitter, mistake_label_column="mistake_category_label"):
    for _, mistake_category_idx in enumerate(task_embeddings_df[mistake_label_column].unique()):
        mistake_category_df = task_embeddings_df[task_embeddings_df[mistake_label_column] == mistake_category_idx]
        marker_color = get_color_for_category(mistake_category_idx)  # Get color for Mistake Category Type
        text_column = "category_hint"

        x_values = mistake_category_df[f'reduced_{embedding_type_prefix}_embedding_1'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_1'].max() - task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_1'].min())
        y_values = mistake_category_df[f'reduced_{embedding_type_prefix}_embedding_2'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_2'].max() - task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_2'].min())

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers', marker=dict(color=marker_color, line=dict(width=1, color='DarkSlateGrey')),
            name=mistake_category_df['mistake_category_name'].iloc[0], text=mistake_category_df[text_column], hoverinfo='text+name'
        ))
    return fig


def build_scatter_plot_with_mistake_category_trace(task_embeddings_df, embedding_type_prefix,  jitter=0.01):
    title = f'Visualizing Clusters of Mistake Labels from {embedding_type_prefix} Embeddings'

    fig = go.Figure()
    fig = add_traces_by_mistake_category(fig, task_embeddings_df, embedding_type_prefix, jitter)

    fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', clickmode='event+select', width=1600, height=800,
                      title={'text': title,'y': 0.9,'x': 0.5,'xanchor': 'center','yanchor': 'top','font': {'size': 20, 'color': 'black', 'family': "Arial"}},
                      legend=dict(# x=1.1,# y=0.5,# xanchor='right',# yanchor='middle',# orientation='v',
                          title=dict(text='Mistake Category', side='top')),)
    return fig


def plot_mistake_statistics(df, category_col='mistake_category_name', category_idx_col='mistake_category_label'):
    """
    Creates a pie chart and a bar chart figure showing the distribution of student mistakes by category.

    Parameters:
        df (pd.DataFrame): DataFrame containing the mistake categories and their counts.
        category_col (str): Column name for the mistake categories.
        category_idx_col (str): Column name for the mistake category indices.

    Returns:
        tuple: A tuple containing the pie chart figure and the bar chart figure.
    """
    # Data preparation: Ensure the DataFrame contains the expected columns and data types
    if category_col not in df.columns:
        raise ValueError(f"The DataFrame must contain the column '{category_col}'.")

    # Count the occurrences of each category
    df_count = df.groupby(category_col).size().reset_index(name='count')

    # Create a consistent color mapping
    color_discrete_map = {category: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, category in enumerate(df_count[category_col])}

    # Generate Pie Chart
    pie_fig = px.pie(df_count, names=category_col, values='count', title='Distribution of Student Mistakes (Pie Chart)',
                     color_discrete_map=color_discrete_map)
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(width=1000, height=1000)

    # Generate Bar Chart
    bar_fig = px.bar(df_count, x=category_col, y='count', title='Distribution of Student Mistakes (Bar Chart)',
                     color=category_col, text='count', color_discrete_map=color_discrete_map)
    bar_fig.update_traces(texttemplate='%{text}', textposition='outside')
    bar_fig.update_layout(xaxis_title='Mistake Category', yaxis_title='Frequency',
                          uniformtext_minsize=8, uniformtext_mode='hide', width=1200, height=800)

    return pie_fig, bar_fig