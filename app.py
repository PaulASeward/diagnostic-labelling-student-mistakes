import dash
from dash import dcc, html, Input, Output, callback, State
from dash import callback_context, dash_table
from dash.exceptions import PreventUpdate

from select_utils import TaskSelector
from dimension_reduction import available_techniques
from plot_utils import *

# Initialize the Dash app
app = dash.Dash(__name__)

task_selector = TaskSelector()

# App layout
app.layout = html.Div([
    html.H1("Data Visualization of TA Corrections"),
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
    # html.Div([
    #     dcc.Checklist(
    #         id='task-checklist',
    #         options=[],
    #         value=[]
    #     )
    # ], style={'margin-bottom': '20px'}),
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'flexWrap': 'nowrap',
                    'margin-bottom': '20px'}, children=[
        html.Div([
            dcc.Checklist(
                id='task-checklist',
                options=[],
                value=[]
            )
        ], style={'flexShrink': '1'}),
        html.Div([
            dcc.Dropdown(
                id='dimension-reduction-technique',
                options=available_techniques(),
                placeholder="Select a Dimension Reduction Technique"
            ),
            html.P("Dimension Reduction Technique", style={'textAlign': 'center'})
        ], style={'flexShrink': '1', 'minWidth': '0', 'width': '50%'})
    ]),
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'flexWrap': 'nowrap',
                    'margin-bottom': '20px'}, children=[
        html.Div([
            html.Button('Generate Dashboards', id='generate-button', n_clicks=0)
        ], style={'flexShrink': '1'}),
        html.Div([
            dcc.Slider(
                id='jitter-slider',
                min=0.000,
                max=0.050,
                step=0.001,
                value=0.001,
                marks={i / 1000: str(i / 1000) for i in range(0, 51, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.P("Jitter Adjustment", style={'textAlign': 'center'})
        ], style={'flexShrink': '1', 'minWidth': '0', 'width': '50%'})
    ]),
    dcc.Graph(id='scatter-plot-by-ta'),
    dash_table.DataTable(
        id='table-by-ta',
        columns=[
            {'name': 'TA Name', 'id': 'ta_name'},
            {'name': 'Student ID', 'id': 'student_id'},
            {'name': 'Grade Difference', 'id': 'grade_difference_percentage'},
            {'name': 'Feedback Sentences Added by TA', 'id': 'feedback_additive_differential'},
            {'name': 'Hyperlink', 'id': 'hyperlink', 'presentation': 'markdown'},
        ],
        markdown_options={"html": True},
        style_table={'width': '70%', 'minWidth': '70%', 'height': 'auto', 'maxHeight': '500px', 'overflowY': 'auto', 'overflowX': 'auto', 'margin': 'auto'},
        style_cell={'fontFamily': 'Arial, sans-serif',  'fontSize': '14px', 'textAlign': 'left', 'whiteSpace': 'normal', 'padding': '10px', 'paddingLeft': '0px', 'minWidth': '100px', 'width': '100px', 'maxWidth': '150px', 'height': 'auto'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f3f3f3', 'color': 'black', 'paddingLeft': '0px', 'borderBottom': '1px solid black'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
    ),
    html.Div([
        dcc.Checklist(id='ta-checklist', options=[], value=[])
    ], style={'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='scatter-plot-by-grade', hoverData=None),
        ], style={'flex': '1', 'min-width': '0', 'padding-right': '0px'}),  # Adjust padding-right as needed
        html.Div([
            dcc.Graph(id='heatmap-feedback-similarity'),
        ], style={'flex': '1', 'min-width': '0', 'padding-left': '0px'}),  # Adjust padding-left as needed
    ], style={'display': 'flex', 'gap': '0px'}),
])


@app.callback(
    Output('assignment-dropdown', 'options'),
    Input('course-dropdown', 'value')
)
def set_assignment_options(selected_course_id):
    task_selector.selections['course'] = selected_course_id
    return [{'label': name, 'value': id} for id, name in task_selector.assignment_mapping.items() if id in task_selector.df_joined[task_selector.df_joined['course_id'] == selected_course_id]['assignment_id'].unique()]


@app.callback(
    Output('task-checklist', 'options'),
    Input('assignment-dropdown', 'value')
)
def set_task_options(selected_assignment_id):
    task_selector.selections['assignment'] = selected_assignment_id
    return [{'label': name, 'value': id} for id, name in task_selector.task_mapping.items() if id in task_selector.df_joined[task_selector.df_joined['assignment_id'] == selected_assignment_id]['task_id'].unique()]


@app.callback(
    [Output('ta-checklist', 'options'),
    Output('ta-checklist', 'value'),
     Output('scatter-plot-by-ta', 'figure'),
     Output('table-by-ta', 'data')],
    [Input('generate-button', 'n_clicks'),
     Input('scatter-plot-by-ta', 'selectedData'),
        Input('dimension-reduction-technique', 'value'),
     Input('jitter-slider', 'value')],
    [State('course-dropdown', 'value'),
     State('assignment-dropdown', 'value'),
     State('task-checklist', 'value'),
     State('scatter-plot-by-ta', 'figure'),
        State('jitter-slider', 'value'),
     State('table-by-ta', 'data')]
)
def update_wholeclass_dashboard(n_clicks, selectedData, dimension_reduction_technique, jitter, selected_course, selected_assignment, selected_tasks, scatter_fig, jitter_state, current_table_fig):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'scatter-plot-by-ta':
        if not selectedData:
            raise PreventUpdate

        selected_points_indices = [point['customdata']['student_id'] for point in selectedData['points']]
        updated_table_data = update_table(selected_points_indices, task_selector.df_with_additive_embeddings)
        return dash.no_update, dash.no_update, dash.no_update, updated_table_data

    elif triggered_id == 'generate-button':
        if n_clicks > 0:
            if selected_course != task_selector.selections['tasks']:  # New tasks are selected

                task_selector.selections['tasks'] = selected_tasks
                task_selector.on_task_selection()

                ta_options = [{'label': ta, 'value': ta} for ta in task_selector.df_with_additive_embeddings['ta_name'].unique()]
                ta_values = [ta['value'] for ta in ta_options]
                # Check for any ta_options
                if not ta_options:
                    ta_options = [{'label': 'No TA options', 'value': 'No TA options'}]
                    ta_values = ['No TA options']

                initial_table_data = []
                #
                # print(f"TA options: {ta_options}")
                # print(f"Task selector Additive Embeddings: {task_selector.df_with_additive_embeddings}")  # Empty DataFrame indicates no columns of embeddings existed in table
                fig1 = None
                if not task_selector.df_with_additive_embeddings.empty:
                    fig1 = build_scatter_plot_with_ta_trace(task_embeddings_df=task_selector.df_with_additive_embeddings, jitter=jitter_state)

                if fig1 is None:
                    fig1 = go.Figure()

                return ta_options, ta_values, fig1, initial_table_data

    elif triggered_id == 'jitter-slider':
        if scatter_fig is not None:
            fig1 = build_scatter_plot_with_ta_trace(task_embeddings_df=task_selector.df_with_additive_embeddings, jitter=jitter)
            return dash.no_update, dash.no_update, fig1, dash.no_update

    elif triggered_id == 'dimension-reduction-technique':
        task_selector.on_dimension_reduction_selection(dimension_reduction_technique)
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    [Output('scatter-plot-by-grade', 'figure'),
     Output('heatmap-feedback-similarity', 'figure')],
    [Input('ta-checklist', 'value'),
     Input('jitter-slider', 'value'),
     Input('heatmap-feedback-similarity', 'hoverData')],
    [State('scatter-plot-by-grade', 'figure'),
     State('ta-checklist', 'value')]
)
def update_ta_dashboard(ta_selections, jitter, hoverData, existing_scatter_fig, existing_ta_selections):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'ta-checklist' or triggered_id == 'jitter-slider':
        scatter_fig = build_scatter_plot_by_grade(task_selector.df_with_additive_embeddings, ta_selections, jitter)
        heatmap_fig = plot_heat_map(task_selector.df_with_additive_embeddings, ta_selections)
        return scatter_fig, heatmap_fig

    elif triggered_id == 'heatmap-feedback-similarity':
        hovered_students = [int(hoverData['points'][0]['x']), int(hoverData['points'][0]['y'])] if hoverData else None
        scatter_fig = build_scatter_plot_by_grade(task_selector.df_with_additive_embeddings, existing_ta_selections, jitter, hovered_students=hovered_students)
        return scatter_fig, dash.no_update

    # Default return if none of the inputs triggered the callback
    return dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)