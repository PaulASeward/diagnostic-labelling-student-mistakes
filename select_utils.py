import ast
import pandas as pd
from plot_utils import *
from embeddings import get_processed_embeddings
from dimension_reduction import project_embeddings_to_reduced_dimension

JOINED_FEEDBACK_PATH = 'data/joined_feedback.csv'
FEEDBACK_DIFFERENTIAL_PATH = 'data/feedback_differential_table.csv'
TA_OVERRIDES_PATH = 'data/ta-overrides.csv'
AI_FEEDBACK_PATH = 'data/ai-feedback.csv'


class TaskSelector:
    def __init__(self, path_to_feedback_diff_data=FEEDBACK_DIFFERENTIAL_PATH, path_to_joined_feedback_data=JOINED_FEEDBACK_PATH):
        self.df_diff = pd.read_csv(path_to_feedback_diff_data)
        self.df_joined = pd.read_csv(path_to_joined_feedback_data)  # Load the larger DataFrame for mappings
        self.course_mapping = self._create_mapping('course_id', 'course_name')
        self.assignment_mapping = self._create_mapping('assignment_id', 'assignment_name')
        self.task_mapping = self._create_mapping('task_id', 'task_title')
        self.selections = {'course': None, 'assignment': None, 'tasks': []}
        self.selected_df = None
        self.df_with_additive_embeddings = None
        self.additive_embedding_array = None
        self.df_with_subtractive_embeddings = None
        self.dimension_reduction_technique = 'PCA'

    def _create_mapping(self, id_column, title_column):
        """Create a mapping from id to title for dropdown options."""
        return self.df_joined[[id_column, title_column]].drop_duplicates().set_index(id_column)[title_column].to_dict()

    def on_dimension_reduction_selection(self, reduction_technique):
        self.dimension_reduction_technique = reduction_technique

    def on_task_selection(self):
        selected_course = self.selections['course']
        selected_assignment = self.selections['assignment']
        selected_tasks = self.selections['tasks']

        if selected_course and selected_assignment and selected_tasks:
            self.selected_df = self.df_diff[(self.df_diff['course_id'] == selected_course) &
                                            (self.df_diff['assignment_id'] == selected_assignment) &
                                            (self.df_diff['task_id'].isin(selected_tasks))]

            df_with_embeddings, self.additive_embedding_array = get_processed_embeddings(self.selected_df, 'additive')
            # Add check that technique is set
            self.df_with_additive_embeddings = project_embeddings_to_reduced_dimension(df_with_embeddings, self.additive_embedding_array, 'additive', self.dimension_reduction_technique)
            if not self.df_with_additive_embeddings.empty:
                self.df_with_additive_embeddings['student_id_str'] = self.df_with_additive_embeddings['student_id'].astype(str)
                self.df_with_additive_embeddings['hyperlink'] = self.df_with_additive_embeddings.apply(lambda row: f"https://app.stemble.ca/web/courses/{row['course_id']}/assignments/{row['assignment_id']}/marking/{row['student_id']}/tasks/{row['task_id']}", axis=1)
                self.df_with_additive_embeddings['grade_difference_percentage'] = self.df_with_additive_embeddings['grade_difference'].apply(lambda x: f"+{(x * 100):.2f}%" if x > 0 else f"{(x * 100):.2f}%")
                self.df_with_additive_embeddings['grade_difference_transformed'] = self.df_with_additive_embeddings['grade_difference'].apply(lambda x: np.sign(x) * np.sqrt(np.abs(x)))  # Signed square root function to scrutinize near zero points
                self.df_with_additive_embeddings[f"feedback_additive_embedding_np"] = self.df_with_additive_embeddings['feedback_additive_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


