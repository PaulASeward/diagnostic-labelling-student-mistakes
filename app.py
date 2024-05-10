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
                    ], style={'flexShrink': '1', 'minWidth': '0', 'width': '100%'}),
                    html.Div([
                        dcc.Dropdown(
                            id='clustering-technique',
                            options=available_clustering_techniques(),
                            placeholder="Select a Clustering Technique"
                        ),
                    ], style={'flexShrink': '1', 'minWidth': '0', 'width': '100%'})
                ]
            ),
        ]
    ),
    html.Div([
        html.Button('Load Data', id='load-button', n_clicks=0),
        html.Button('Generate Dashboard', id='generate-button', n_clicks=0),
    ]),
    html.Div([
        dcc.Graph(id='scatter-plot')
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='pie_fig'),
        ], style={'flex': '1', 'min-width': '0', 'padding-right': '0px'}),  # Adjust padding-right as needed
        html.Div([
            dcc.Graph(id='bar_fig'),
        ], style={'flex': '1', 'min-width': '0', 'padding-left': '0px'}),  # Adjust padding-left as needed
    ], style={'display': 'flex', 'gap': '0px'}),
    html.Div(id='dummy-output', style={'display': 'none'}),
])


@app.callback(
    Output('dummy-output', 'children'),
    Input('load-button', 'n_clicks'),
    [State('course-dropdown', 'value'),
     State('assignment-dropdown', 'value'),
     State('task-checklist', 'value')]
)
def choose_tasks(n_clicks, selected_course, selected_assignment, selected_tasks):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'load-button':
        if n_clicks > 0:
            if selected_tasks != task_selector.selections['tasks']:  # New tasks are selected
                task_selector.selections['tasks'] = selected_tasks
                task_selector.on_task_selection()
    return dash.no_update

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
    [Output('scatter-plot', 'figure'),
     Output('pie_fig', 'figure'),
     Output('bar_fig', 'figure')],
    [Input('generate-button', 'n_clicks'),
     Input('dimension-reduction-technique', 'value'),
     Input('clustering-technique', 'value'),],
    [State('course-dropdown', 'value'),
     State('assignment-dropdown', 'value'),
     State('task-checklist', 'value')]
)
def update_wholeclass_dashboard(n_clicks, dimension_reduction_technique, clustering_technique, selected_course, selected_assignment, selected_tasks):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'generate-button':
        if n_clicks > 0:
            task_selector.cluster_and_categorize()
            fig1 = go.Figure()
            pie_fig = go.Figure()
            bar_fig = go.Figure()
            if not task_selector.df_with_category_embeddings.empty:
                fig1 = build_scatter_plot_with_mistake_category_trace(task_embeddings_df=task_selector.df_with_category_embeddings, embedding_type_prefix='category_hint')
                pie_fig, bar_fig = plot_mistake_statistics(task_selector.df_with_category_embeddings)

            return fig1, pie_fig, bar_fig

    elif triggered_id == 'dimension-reduction-technique':
        task_selector.on_dimension_reduction_selection(dimension_reduction_technique)
        return dash.no_update, dash.no_update, dash.no_update
    elif triggered_id == 'clustering-technique':
        task_selector.on_clustering_technique_selection(clustering_technique)
        return dash.no_update, dash.no_update, dash.no_update


    return dash.no_update, dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)