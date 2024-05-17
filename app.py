import dash
from dash import dcc, html, Input, Output, callback, State
from dash import callback_context, dash_table
from dash.exceptions import PreventUpdate

from select_utils import TaskSelector
from dimension_reduction import available_dimension_reduction_techniques
from clustering_utils import available_clustering_techniques
from plot_utils import *

# Initialize the Dash app
app = dash.Dash(__name__)

task_selector = TaskSelector()

# App layout
app.layout = html.Div([
    html.H1("Diagnostic Labelling of Student Mistakes"),
    html.Div([
        dcc.Dropdown(
            id='course-dropdown',
            placeholder="Select a course",
            options=[{'label': name, 'value': id} for id, name in task_selector.course_mapping.items()],
        )
    ], style={'margin-bottom': '20px'}),
    html.Div([
        dcc.Dropdown(
            id='assignment-dropdown',
            options=[],
            placeholder="Select an assignment"
        )
    ], style={'margin-bottom': '20px'}),
    html.Div(
        style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'flexWrap': 'nowrap',
            'margin-bottom': '20px'
        },
        children=[
            html.Div([
                dcc.Checklist(
                    id='task-checklist',
                    options=[],
                    value=[]
                )
            ], style={'flexGrow': '1', 'flexBasis': '50%'}),
            html.Div(
                style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'justifyContent': 'space-between',
                    'alignItems': 'center',
                    'margin-bottom': '20px',
                    'flexGrow': '1',
                    'flexBasis': '50%'
                },
                children=[
                    html.Div([
                        dcc.Dropdown(
                            id='dimension-reduction-technique',
                            options=available_dimension_reduction_techniques(),
                            placeholder="Select a Dimension Reduction Technique"
                        ),
                    ], style={'flexShrink': '1', 'minWidth': '0', 'width': '100%', 'padding-top': '4px', 'padding-bottom': '6px'}),
                    html.Div([
                        dcc.Dropdown(
                            id='clustering-technique',
                            options=available_clustering_techniques(),
                            placeholder="Select a Clustering Technique"
                        ),
                    ], style={'flexShrink': '1', 'minWidth': '0', 'width': '100%', 'padding-top': '4px', 'padding-bottom': '6px'}),
                    html.Div([
                        dcc.RadioItems(
                            id='mistake-selection-mode',
                            options=[
                                {'label': 'Single Mistake Label', 'value': 'single'},
                                {'label': 'Multiple Mistake Labels', 'value': 'multiple'}
                            ],
                            value='multiple',
                            labelStyle={'display': 'inline-block'}
                        )
                    ], style={'flexShrink': '1', 'minWidth': '0', 'width': '100%', 'padding-top': '8px', 'padding-bottom': '6px'}),
                ]
            ),
        ]
    ),
    html.Div([
        html.Div([
            html.Button('Load Data', id='load-button', n_clicks=0),
        ], style={'padding-right': '8px'}),
        html.Div([
            html.Button('Generate Dashboard', id='generate-button', n_clicks=0),
        ], style={'padding-right': '8px'}),
        html.Div([
            dcc.Dropdown(
                id='cluster-groups-dropdown',
                placeholder="Select number of clusters",
                options=[
                    {'label': 'Auto (Default)', 'value': -1},
                     {'label': '1', 'value': 1},
                     {'label': '2', 'value': 2},
                     {'label': '3', 'value': 3},
                     {'label': '4', 'value': 4},
                     {'label': '5', 'value': 5},
                     {'label': '6', 'value': 6},
                     {'label': '7', 'value': 7},
                     {'label': '8', 'value': 8},
                    {'label': '9', 'value': 9},
                    {'label': '10', 'value': 10},
                ],
            ),
        ], style={'width': '25%', 'padding-right': '8px'}),
        html.Div([
            dcc.Dropdown(
                id='manual-selection_override-dropdown',
                placeholder="Allow Manual Overrides of Category Labels",
                options=[
                    {'label': 'Auto (Default): Labels and Clustered are automatically generated', 'value': 0},
                     {'label': 'Manual Label Suggestion: Cluster algorithms are initialized with suggested center.', 'value': 1},
                     {'label': 'Manual Label Override: No clustering: Closest mistake label is selected (Euclidean distance).', 'value': 2},
                ],
            ),
        ], style={'width': '25%', 'padding-right': '0px'})
    ], style={'display': 'flex', 'flexGrow': '1', 'gap': '8px'}),
    html.Div([
        html.Div([
            dash_table.DataTable(
                id='manual-mistake-label-table',
                columns=[
                    {'name': 'Mistake Category', 'id': 'option', 'editable': True}
                ],
                data=[],
                editable=True,
                row_deletable=True,
                style_table={'width': '33%', 'minWidth': '33%'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f3f3f3'},
                style_cell={
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '14px',
                    'textAlign': 'left',
                    'padding': '10px',
                }
            ),
        ], style={'padding-right': '2px'}),
        html.Div([
            html.Button('Add Category', id='adding-rows-button', n_clicks=0),
        ], style={'padding-top': '10px', 'padding-bottom': '10px'}),
    ], style={'margin-bottom': '0px', 'display': 'flex', 'flexGrow': '1', 'gap': '25px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='scatter-plot', hoverData=None),
        ], style={'flex': '1', 'min-width': '0', 'width': '70%', 'padding-right': '2px'}),
        html.Div([
            dcc.Graph(id='pie_fig'),
        ], style={'flex': '1', 'min-width': '0', 'padding-left': '4px'}),
    ], style={'display': 'flex', 'flexGrow': '1', 'gap': '150px'}),
    dash_table.DataTable(
        id='table-feedback',
        columns=[
            {'name': 'Clustered Mistake Area Label', 'id': 'mistake_category_name'},
            {'name': 'Suggested Mistake Area', 'id': 'category_hint'},
            {'name': 'Entire Feedback', 'id': 'ta_feedback_text'},
            {'name': 'Other Mistake Area Suggestions', 'id': 'category_hints'},
            {'name': 'Grade', 'id': 'formatted_grade', 'presentation': 'markdown'},
        ],
        markdown_options={"html": True},
        style_table={'width': '70%', 'minWidth': '70%', 'height': 'auto', 'maxHeight': '500px', 'overflowY': 'auto',
                     'overflowX': 'auto', 'margin': 'auto'},
        style_cell={'fontFamily': 'Arial, sans-serif', 'fontSize': '14px', 'textAlign': 'left', 'whiteSpace': 'normal',
                    'padding': '10px', 'paddingLeft': '0px', 'minWidth': '100px', 'width': '100px', 'maxWidth': '150px',
                    'height': 'auto'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f3f3f3', 'color': 'black', 'paddingLeft': '0px',
                      'borderBottom': '1px solid black'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                                {'if': {'column_id': 'ta_feedback_text'}, 'textAlign': 'left', 'padding': '15px', 'fontSize': '16px', 'minWidth': '300px', 'width': '40%', 'maxWidth': '600px'}]
    ),
    html.Div([
        html.Div([
        html.P("Labels Displayed:", style={'textAlign': 'left', 'fontWeight': 'bold',}),
        dcc.RadioItems(
            id='dendrogram_label',
            options=[
                {'label': 'Suggested Category', 'value': CATEGORY_HINT_COL},
                {'label': 'Clustered Group', 'value': CATEGORY_NAME_COL},
            ],
            value=CATEGORY_NAME_COL,
            labelStyle={'display': 'inline-block'}
        )]),
        html.Div([
            html.P("Colours Displayed:", style={'textAlign': 'left', 'fontWeight': 'bold',}),
            dcc.RadioItems(
                id='dendrogram_color',
                options=[
                    {'label': 'Above Algorithm Clusters', 'value': True},
                    {'label': 'Dendrogram Clusters', 'value': False},
                ],
                value=False,
                labelStyle={'display': 'inline-block'}
            )])
    ], style={'display': 'flex', 'flexGrow': '1', 'gap': '25px'}),
    html.Div([
        dcc.Graph(id='dendro_fig'),
    ], style={'margin-bottom': '20px'}),
    html.Div(id='dummy-output', style={'display': 'none'}),
])


@app.callback(
    Output('assignment-dropdown', 'options'),
    Input('course-dropdown', 'value')
)
def set_assignment_options(selected_course_id):
    task_selector.selections['course'] = selected_course_id
    return [{'label': name, 'value': id} for id, name in task_selector.assignment_mapping.items() if id in task_selector.df_feedback[task_selector.df_feedback['course_id'] == selected_course_id]['assignment_id'].unique()]


@app.callback(
    Output('task-checklist', 'options'),
    Input('assignment-dropdown', 'value')
)
def set_task_options(selected_assignment_id):
    task_selector.selections['assignment'] = selected_assignment_id
    return [{'label': name, 'value': id} for id, name in task_selector.task_mapping.items() if id in task_selector.df_feedback[task_selector.df_feedback['assignment_id'] == selected_assignment_id]['task_id'].unique()]


@app.callback(
    [Output('manual-mistake-label-table', 'data'),
     Output('scatter-plot', 'figure'),
     Output('pie_fig', 'figure'),
     Output('dendro_fig', 'figure'),
     Output('table-feedback', 'data'),
     Output('dimension-reduction-technique', 'value'),
     Output('clustering-technique', 'value'),
     Output('cluster-groups-dropdown', 'value'),
     Output('manual-selection_override-dropdown', 'value')],
    [Input('adding-rows-button', 'n_clicks'),
     Input('generate-button', 'n_clicks'),
     Input('load-button', 'n_clicks'),
     Input('scatter-plot', 'selectedData'),
     Input('dimension-reduction-technique', 'value'),
     Input('clustering-technique', 'value'),
     Input('cluster-groups-dropdown', 'value'),
     Input('manual-selection_override-dropdown', 'value'),
     Input('mistake-selection-mode', 'value'),
    Input('dendrogram_label', 'value'),
     Input('dendrogram_color', 'value')],
    [State('course-dropdown', 'value'),
     State('assignment-dropdown', 'value'),
     State('task-checklist', 'value'),
     State('manual-mistake-label-table', 'data'),
     State('manual-mistake-label-table', 'columns')]
)
def update_dashboard(n_clicks_add, n_clicks_generate, n_clicks_load, selected_data, dimension_reduction_technique, clustering_technique, n_clusters, manual_override, mistake_selection, dendrogram_label, dendrogram_color, selected_course, selected_assignment, selected_tasks, mistake_table_current_data, mistake_table_columns):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'generate-button':
        if n_clicks_generate > 0:
            if mistake_table_current_data and len(mistake_table_current_data) > 0:
                manually_selected_categories = task_selector.on_manual_categories_selection(mistake_table_current_data)

            task_selector.on_clustering_request()
            task_selector.on_dim_reduction_request()

            fig1 = go.Figure()
            pie_fig = go.Figure()
            dendro_fig = go.Figure()
            initial_table_data = []
            suggested_mistake_categories = [{'option': name} for name in task_selector.cluster_algorithm.mistake_categories_dict.keys()]

            if not task_selector.df_with_category_embeddings.empty:
                task_selector.color_map = create_color_map(task_selector.df_with_category_embeddings, task_selector.cluster_algorithm.mistake_categories_dict)
                fig1 = build_scatter_plot_with_mistake_category_trace(task_selector.df_with_category_embeddings, task_selector.cluster_algorithm.mistake_categories_dict, task_selector.color_map)
                pie_fig = plot_mistake_statistics(task_selector.df_with_category_embeddings, task_selector.cluster_algorithm.mistake_categories_dict, task_selector.color_map)
                dendro_fig = plot_dendrogram(task_selector.df_with_category_embeddings, task_selector.cluster_algorithm.mistake_categories_dict, task_selector.color_map, task_selector.dendrogram_label)

            return suggested_mistake_categories, fig1, pie_fig, dendro_fig, initial_table_data, task_selector.dimension_reduction_technique, task_selector.cluster_algorithm.clustering_technique, task_selector.cluster_algorithm.n_clusters, task_selector.cluster_algorithm.use_manual_mistake_categories

    elif triggered_id == 'adding-rows-button':
        if n_clicks_add > 0:
            mistake_table_current_data.append({c['id']: '' for c in mistake_table_columns})
        return mistake_table_current_data, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, len(mistake_table_current_data), dash.no_update

    elif triggered_id == 'load-button':
        if n_clicks_load > 0:
            if selected_tasks != task_selector.selections['tasks']:  # New tasks are selected
                task_selector.selections['tasks'] = selected_tasks
                task_selector.on_task_selection()
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    elif triggered_id == 'scatter-plot':
        if not selected_data:
            raise PreventUpdate
        selected_points_indices = [(point['customdata']['student_id'], point['customdata']['category_hint_idx']) for point in selected_data['points']]
        updated_table_data = task_selector.update_table(selected_points_indices, task_selector.df_with_category_embeddings)
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, updated_table_data, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    elif triggered_id == 'dimension-reduction-technique':
        task_selector.dimension_reduction_technique = dimension_reduction_technique
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif triggered_id == 'clustering-technique':
        task_selector.cluster_algorithm.clustering_technique = clustering_technique
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update if clustering_technique != 'DBSCAN' else 0
    elif triggered_id == 'cluster-groups-dropdown':
        task_selector.cluster_algorithm.n_clusters = n_clusters
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif triggered_id == 'manual-selection_override-dropdown':
        task_selector.cluster_algorithm.use_manual_mistake_categories = manual_override
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, len(mistake_table_current_data), dash.no_update
    elif triggered_id == 'mistake-selection-mode':
        task_selector.number_mistake_labels = 3 if mistake_selection == 'multiple' else 1
    elif triggered_id == 'dendrogram_label':
        task_selector.dendrogram_label = dendrogram_label
        dendro_fig = plot_dendrogram(task_selector.df_with_category_embeddings, task_selector.cluster_algorithm.mistake_categories_dict, task_selector.color_map, task_selector.dendrogram_label, task_selector.dendrogram_color)
        return dash.no_update, dash.no_update, dash.no_update, dendro_fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif triggered_id == 'dendrogram_color':
        task_selector.dendrogram_color = dendrogram_color
        dendro_fig = plot_dendrogram(task_selector.df_with_category_embeddings, task_selector.cluster_algorithm.mistake_categories_dict, task_selector.color_map, task_selector.dendrogram_label, task_selector.dendrogram_color)
        return dash.no_update, dash.no_update, dash.no_update, dendro_fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)
