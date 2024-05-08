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
    html.Div([
        dcc.Checklist(
            id='task-checklist',
            options=[],
            value=[]
        )
    ], style={'margin-bottom': '20px'}),
    html.Button('Generate', id='generate-button', n_clicks=0),
    html.Div([
        dcc.Graph(id='scatter-plot')
    ])
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
    [Output('scatter-plot', 'figure')],
    [Input('generate-button', 'n_clicks')],
    [State('course-dropdown', 'value'),
     State('assignment-dropdown', 'value'),
     State('task-checklist', 'value')]
)
def update_wholeclass_dashboard(n_clicks, selected_course, selected_assignment, selected_tasks):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'generate-button':
        if n_clicks > 0:
            if selected_course != task_selector.selections['tasks']:  # New tasks are selected

                task_selector.selections['tasks'] = selected_tasks
                task_selector.on_task_selection()

                fig1 = None
                if not task_selector.selected_df.empty:
                    # fig1 = build_scatter_plot(df=task_selector.selected_df)
                    x=1

                if fig1 is None:
                    fig1 = go.Figure()

                return fig1

    return dash.no_update



if __name__ == '__main__':
    app.run_server(debug=True)