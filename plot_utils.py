import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
import plotly.express as px


def get_color_for_ta(index, color_palette=px.colors.qualitative.Plotly):
    return color_palette[index % len(color_palette)]


def add_traces_by_ta(fig, task_embeddings_df, color_design, diff_type_prefix, jitter):
    for i, ta_id in enumerate(task_embeddings_df['ta_id'].unique()):
        ta_filtered_df = task_embeddings_df[task_embeddings_df['ta_id'] == ta_id]
        marker_color = get_color_for_ta(i, color_design)  # Get color for TA

        custom_data = ta_filtered_df.apply(lambda row: {'student_id': row['student_id'], 'grade_difference': row['grade_difference']}, axis=1).tolist()
        ta_name = ta_filtered_df['ta_name'].iloc[0] if not ta_filtered_df['ta_name'].empty else "Unknown TA"

        # Add grade_difference_percentage to the text column for display in the hover tooltip
        text = ta_filtered_df[f'feedback_{diff_type_prefix}_differential'] + '<br>' + 'Grade Change: ' + ta_filtered_df['grade_difference_percentage']  + 'Student Id: ' + ta_filtered_df['student_id_str']

        x_values = ta_filtered_df[f'reduced_{diff_type_prefix}_embedding_1'] + (np.random.rand(len(ta_filtered_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_additive_embedding_1'].max() - task_embeddings_df[f'reduced_additive_embedding_1'].min())
        y_values = ta_filtered_df[f'reduced_{diff_type_prefix}_embedding_2'] + (np.random.rand(len(ta_filtered_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_additive_embedding_2'].max() - task_embeddings_df[f'reduced_additive_embedding_2'].min())

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers', marker=dict(color=marker_color, line=dict(width=1, color='DarkSlateGrey')),
            name=ta_name, text=text, customdata=custom_data, hoverinfo='text+name'
        ))
    return fig


def build_scatter_plot_with_ta_trace(task_embeddings_df, jitter=0.01):
    title = 'Visualizing TA Feedback Trends'

    fig = go.Figure()
    color_palette = px.colors.qualitative.Plotly

    fig = add_traces_by_ta(fig, task_embeddings_df, color_palette, 'additive', jitter)

    fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', clickmode='event+select', width=1600, height=800,
                      title={'text': title,'y': 0.9,'x': 0.5,'xanchor': 'center','yanchor': 'top','font': {'size': 20, 'color': 'black', 'family': "Arial"}},
                      legend=dict(# x=1.1,# y=0.5,# xanchor='right',# yanchor='middle',# orientation='v',
                          title=dict(text='TA', side='top')),)
    return fig


def update_table(current_selection_indices, task_embeddings_df):
    filtered_df = task_embeddings_df[task_embeddings_df['student_id'].isin(current_selection_indices)]
    filtered_df['hyperlink'] = filtered_df['hyperlink'].apply(lambda x: f"[Link]({x})")
    filtered2_df = filtered_df[['ta_name', 'student_id', 'grade_difference_percentage', 'feedback_additive_differential', 'hyperlink']]
    data_to_display = filtered2_df.to_dict('records')
    return data_to_display


def build_scatter_plot_by_grade(task_embeddings_df, ta_selections=None, jitter=0.01, hovered_students=None):
    title = 'TA Grade Change Patterns'
    custom_color_scale = [[0.0, "red"], [0.5, "white"], [1.0, "blue"]]

    if ta_selections is not None:
        task_embeddings_df = task_embeddings_df[task_embeddings_df['ta_name'].isin(ta_selections)]

    task_embeddings_df['marker_size'] = 8
    if hovered_students is not None:
        is_hovered = task_embeddings_df['student_id'].isin(hovered_students)
        task_embeddings_df['marker_size'] = is_hovered.apply(lambda x: 20 if x else 8)
    else:
        task_embeddings_df['marker_size'] = 8

    if jitter is not None:
        np.random.seed(42)
        task_embeddings_df[f'reduced_additive_embedding_1'] += (np.random.rand(len(task_embeddings_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_additive_embedding_1'].max() - task_embeddings_df[f'reduced_additive_embedding_1'].min())
        task_embeddings_df[f'reduced_additive_embedding_2'] += (np.random.rand(len(task_embeddings_df)) - 0.5) * jitter * (task_embeddings_df[f'reduced_additive_embedding_2'].max() - task_embeddings_df[f'reduced_additive_embedding_2'].min())

    hover_data = {column: False for column in task_embeddings_df.columns}
    hover_data['feedback_additive_differential'] = True
    hover_data['grade_difference_percentage'] = True

    custom_data = task_embeddings_df.apply(lambda row: {'student_id': row['student_id']}, axis=1).tolist()
    fig = go.Figure(data=[go.Scatter(
        x=task_embeddings_df['reduced_additive_embedding_1'],
        y=task_embeddings_df['reduced_additive_embedding_2'],
        customdata=custom_data, mode='markers',
        text=task_embeddings_df['feedback_additive_differential'] + '<br>' + 'Grade Change: ' + task_embeddings_df['grade_difference_percentage'] + ' Student Id: ' + task_embeddings_df['student_id_str'],
        hoverinfo='text',
        marker=dict(
            color=task_embeddings_df['grade_difference'],
            size=task_embeddings_df['marker_size'],
            colorscale=custom_color_scale,
            coloraxis='coloraxis',
        ))])

    fig.update_layout(
        title={'text': title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
               'font': {'size': 20, 'color': 'black', 'family': "Arial"}},
        xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', width=800, height=800,
        coloraxis=dict(
            colorscale=custom_color_scale,
            cmin=-task_embeddings_df['grade_difference'].abs().max(),
            cmax=task_embeddings_df['grade_difference'].abs().max()
        ),
    )

    return fig


def plot_heat_map(task_embedding_df, ta_selections=None):
    task_embedding_df = task_embedding_df if ta_selections is None else task_embedding_df[task_embedding_df['ta_name'].isin(ta_selections)]
    embedding_array = np.stack(task_embedding_df['feedback_additive_embedding_np'].values)
    distance_matrix = calculate_distance_matrix(embedding_array)
    selected_students = task_embedding_df['student_id'].values

    fig = go.Figure(data=go.Heatmap(
        x=['{}'.format(i) for i in selected_students], y=['{}'.format(i) for i in selected_students],
        hoverinfo="x+y",
        z=distance_matrix, colorscale='Sunset_r', hoverongaps=False))

    fig.update_layout(title={'text': 'Heatmap of Feedback Similarity','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top','font': {'size': 20, 'color': 'black', 'family': "Arial"}},
        xaxis_nticks=36, yaxis_nticks=36, width=800, height=800)
    return fig


def calculate_distance_matrix(embedding_array):
    distances = pdist(embedding_array, 'euclidean')
    return squareform(distances)