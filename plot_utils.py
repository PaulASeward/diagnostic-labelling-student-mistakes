import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
import plotly.express as px

COLOR_PALETTE = px.colors.qualitative.Plotly
CATEGORY_NAME_COL = 'mistake_category_name'
CATEGORY_IDX_COL = 'mistake_category_label'


def create_color_map(task_embeddings_df, color_palette=COLOR_PALETTE):
    unique_indices = task_embeddings_df[CATEGORY_IDX_COL].unique()
    return {idx: color_palette[idx % len(color_palette)] for idx in unique_indices}


def get_color_for_category(mistake_category_idx, color_palette=COLOR_PALETTE):
    return color_palette[mistake_category_idx % len(color_palette)]


def update_table(current_selection_indices, task_embeddings_df):
    filtered_df = task_embeddings_df[task_embeddings_df.apply(lambda row: (row['student_id'], row['category_hint_idx']) in current_selection_indices, axis=1)]
    filtered_df['formatted_grade'] = filtered_df.apply(lambda row: f"[{(row['grade'] * 100):.2f}%](https://app.stemble.ca/web/courses/{row['course_id']}/assignments/{row['assignment_id']}/marking/{row['student_id']}/tasks/{row['task_id']})", axis=1)
    filtered2_df = filtered_df[[CATEGORY_NAME_COL, 'category_hint', 'ta_feedback_text', 'category_hints', 'formatted_grade']]
    data_to_display = filtered2_df.to_dict('records')
    return data_to_display


def add_traces_by_mistake_category(fig, task_embeddings_df, mistake_categories_dict, color_map, jitter, mistake_label_column="mistake_category_label", text_column="category_hint"):
    for _, mistake_category_idx in enumerate(task_embeddings_df[CATEGORY_IDX_COL].unique()):
        mistake_category_df = task_embeddings_df[task_embeddings_df[CATEGORY_IDX_COL] == mistake_category_idx]

        marker_color = color_map[mistake_category_idx]  # Get color for Mistake Category Type
        # marker_color = get_color_for_category(mistake_category_idx)  # Get color for Mistake Category Type
        custom_data = mistake_category_df.apply(lambda row: {'student_id': row['student_id'], 'category_hint_idx': row['category_hint_idx']}, axis=1).tolist()

        x_values = mistake_category_df[f'reduced_category_hint_embedding_1'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_category_hint_embedding_1'].max() - task_embeddings_df[f'reduced_category_hint_embedding_1'].min())
        y_values = mistake_category_df[f'reduced_category_hint_embedding_2'] + (np.random.rand(len(mistake_category_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_category_hint_embedding_2'].max() - task_embeddings_df[f'reduced_category_hint_embedding_2'].min())

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers', marker=dict(color=marker_color, line=dict(width=1, color='DarkSlateGrey')),
            name=mistake_category_df[CATEGORY_NAME_COL].iloc[0], text=mistake_category_df[text_column], customdata=custom_data, hoverinfo='text+name'
        ))
    return fig


def build_scatter_plot_with_mistake_category_trace(task_embeddings_df, mistake_categories_dict, jitter=0.02):
    title = f'Visualizing Clusters of Mistake Labels from Reduced Embeddings'
    color_map = create_color_map(task_embeddings_df)

    fig = go.Figure()
    fig = add_traces_by_mistake_category(fig, task_embeddings_df, mistake_categories_dict, color_map, jitter)

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
    df_count['color'] = df_count[CATEGORY_IDX_COL].apply(lambda idx: get_color_for_category(idx))
    color_discrete_map = {row[CATEGORY_NAME_COL]: row['color'] for index, row in df_count.iterrows()}

    # Generate Pie Chart
    pie_fig = px.pie(df_count, names=CATEGORY_NAME_COL, values='count', title='Distribution of Student Mistakes (Pie Chart)',
                     color_discrete_map=color_discrete_map)
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(width=800, height=800)

    return pie_fig