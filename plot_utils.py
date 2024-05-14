import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
import plotly.express as px

COLOR_PALETTE = px.colors.qualitative.Plotly


def create_color_map(task_embeddings_df, color_palette=COLOR_PALETTE):
    unique_indices = task_embeddings_df['mistake_category_label'].unique()
    return {idx: color_palette[idx % len(color_palette)] for idx in unique_indices}


def get_color_for_category(mistake_category_idx, color_palette=COLOR_PALETTE):
    return color_palette[mistake_category_idx % len(color_palette)]


def update_table(current_selection_indices, task_embeddings_df):
    filtered_df = task_embeddings_df[task_embeddings_df.apply(lambda row: (row['student_id'], row['category_hint_idx']) in current_selection_indices, axis=1)]
    filtered_df['formatted_grade'] = filtered_df.apply(lambda row: f"[{(row['grade'] * 100):.2f}%](https://app.stemble.ca/web/courses/{row['course_id']}/assignments/{row['assignment_id']}/marking/{row['student_id']}/tasks/{row['task_id']})", axis=1)
    filtered2_df = filtered_df[['mistake_category_name', 'category_hint', 'ta_feedback_text', 'category_hints', 'formatted_grade']]
    data_to_display = filtered2_df.to_dict('records')
    return data_to_display


def add_traces_by_mistake_category(fig, task_embeddings_df, color_map, embedding_type_prefix, jitter, mistake_label_column="mistake_category_label", text_column="category_hint"):
    for _, mistake_category_idx in enumerate(task_embeddings_df[mistake_label_column].unique()):
        mistake_category_df = task_embeddings_df[task_embeddings_df[mistake_label_column] == mistake_category_idx]

        marker_color = color_map[mistake_category_idx]  # Get color for Mistake Category Type
        # marker_color = get_color_for_category(mistake_category_idx)  # Get color for Mistake Category Type
        custom_data = mistake_category_df.apply(lambda row: {'student_id': row['student_id'], 'category_hint_idx': row['category_hint_idx']}, axis=1).tolist()

        x_values = mistake_category_df[f'reduced_{embedding_type_prefix}_embedding_1'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_1'].max() - task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_1'].min())
        y_values = mistake_category_df[f'reduced_{embedding_type_prefix}_embedding_2'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_2'].max() - task_embeddings_df[f'reduced_{embedding_type_prefix}_embedding_2'].min())

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers', marker=dict(color=marker_color, line=dict(width=1, color='DarkSlateGrey')),
            name=mistake_category_df['mistake_category_name'].iloc[0], text=mistake_category_df[text_column], customdata=custom_data, hoverinfo='text+name'
        ))
    return fig


def build_scatter_plot_with_mistake_category_trace(task_embeddings_df, embedding_type_prefix,  jitter=0.01):
    title = f'Visualizing Clusters of Mistake Labels from {embedding_type_prefix} Embeddings'
    color_map = create_color_map(task_embeddings_df)

    fig = go.Figure()
    fig = add_traces_by_mistake_category(fig, task_embeddings_df, color_map, embedding_type_prefix, jitter)

    fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', clickmode='event+select', width=1200, height=800,
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
        figure: The pie chart figure.
    """
    if category_col not in df.columns or category_idx_col not in df.columns:
        raise ValueError(f"The DataFrame must contain the columns '{category_col}' and '{category_idx_col}'.")

        # Count the occurrences of each category
    df_count = df.groupby([category_col, category_idx_col]).size().reset_index(name='count')
    df_count['color'] = df_count[category_idx_col].apply(lambda idx: get_color_for_category(idx))
    color_discrete_map = {row[category_col]: row['color'] for index, row in df_count.iterrows()}

    # Generate Pie Chart
    pie_fig = px.pie(df_count, names=category_col, values='count', title='Distribution of Student Mistakes (Pie Chart)',
                     color_discrete_map=color_discrete_map)
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(width=800, height=800)

    return pie_fig