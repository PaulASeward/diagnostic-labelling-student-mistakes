import ast
import pandas as pd
from ta_feedback_analysis.plot_utils import *
from ta_feedback_analysis.embeddings import get_processed_embeddings
from ta_feedback_analysis.dimension_reduction import project_embeddings_to_reduced_dimension
import os
import json

AI_FEEDBACK_DATA_NAME = 'ai-feedback.csv'
TA_OVERRIDES_DATA_NAME = 'ta-overrides.csv'
JOINED_FEEDBACK_PATH = 'joined_feedback2.csv'

# TA_FEEDBACK_DATA_NAME = 'ta-feedback-data.csv'
TA_FEEDBACK_DATA_NAME = 'ta-feedback-data2.csv'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEEDBACK_PATH = os.path.join(DATA_DIR, TA_FEEDBACK_DATA_NAME)


AI_FEEDBACK_DATA_PATH = os.path.join(DATA_DIR, AI_FEEDBACK_DATA_NAME)
TA_OVERRIDES_DATA_PATH = os.path.join(DATA_DIR, TA_OVERRIDES_DATA_NAME)
JOINED_FEEDBACK_PATH = os.path.join(DATA_DIR, JOINED_FEEDBACK_PATH)

# df = pd.read_csv(JOINED_FEEDBACK_PATH)
# df.loc[df['course_id'] == 1540, 'course_name'] = "General Chemistry Laboratory I"
# df.loc[df['course_id'] == 1565, 'course_name'] = "General Chemistry Laboratory II"
# df.to_csv(JOINED_FEEDBACK_PATH, index=False)


class TaskSelector:
    def __init__(self, path_to_feedback_diff_data=FEEDBACK_PATH):
        self.df_diff = pd.read_csv(path_to_feedback_diff_data)
        self.df_joined = pd.read_csv(JOINED_FEEDBACK_PATH)  # Load the larger DataFrame for mappings
        # Remove any instances of df_joined with NaN values in the course_id, assignment_id, or task_id columns
        self.df_joined.dropna(subset=['course_name'], inplace=True)


        self.course_mapping = self._create_mapping('course_id', 'course_name')
        self.assignment_mapping = self._create_mapping('assignment_id', 'assignment_name')
        self.task_mapping = self._create_mapping('task_id', 'task_title')
        self.selections = {'course': None, 'assignment': None, 'tasks': []}
        self.selected_df = None
        self.df_with_additive_embeddings = None
        self.additive_embedding_array = None
        self.dimension_reduction_technique = 'PCA'

    def _create_mapping(self, id_column, title_column):
        """Create a mapping from id to title for dropdown options."""
        sorted_df = self.df_joined[[id_column, title_column]].drop_duplicates().sort_values(by=title_column, ascending=True)
        return sorted_df.set_index(id_column)[title_column].to_dict()

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


def select_questions_to_process(feedback_df, selected_courses=None, selected_assignments=None, selected_tasks=None, selected_parts=None):
    if selected_courses is not None:
        if all(isinstance(item, (int, np.integer)) for item in selected_courses):
            feedback_df = feedback_df[feedback_df['course_id'].isin(selected_courses)]
        else:
            feedback_df = feedback_df[feedback_df['course_name'].isin(selected_courses)]

    if selected_assignments is not None:
        if all(isinstance(item, (int, np.integer)) for item in selected_assignments):
            feedback_df = feedback_df[feedback_df['assignment_id'].isin(selected_assignments)]
        else:
            feedback_df = feedback_df[feedback_df['assignment_name'].isin(selected_assignments)]

    if selected_tasks is not None:
        if all(isinstance(item, (int, np.integer)) for item in selected_tasks):
            feedback_df = feedback_df[feedback_df['task_id'].isin(selected_tasks)]
        else:
            feedback_df = feedback_df[feedback_df['task_title'].isin(selected_tasks)]

    if selected_parts is not None:
        feedback_df = feedback_df[feedback_df['part_name'].isin(selected_parts)]

    return feedback_df


def get_ai_grade(student_grades_group, part_name=None):
    if part_name is None:
       return student_grades_group['ai_grade'].iloc[0] if not np.isnan(student_grades_group['ai_grade'].iloc[0]) else 0
    else:
        return 0  # Eventually, connect to Grade getion by Part


def get_ta_grade(student_grades_group, part_name=None):
    if part_name is None:
        return student_grades_group['ta_grade'].iloc[0] if not np.isnan(student_grades_group['ta_grade'].iloc[0]) else 0
    else:
        return 0  # Eventually, Connect to Grade getion by Part


def get_single_ai_feedback_for_part(student_part_response_group, part_name=None):
    feedback_data = student_part_response_group['feedback_data'].iloc[0]
    if pd.isna(feedback_data):
        return ''

    try:
        feedback_json = json.loads(feedback_data)
        if 'feedback' in feedback_json:
            return feedback_json['feedback']
        else:
            return ''
    except json.JSONDecodeError as e:
        print(f'Error: {e}')
        return ''


def get_ta_feedback(student_response_group, part_name=None):
    if part_name is None:
            feedback = student_response_group['ta_feedback_text'].iloc[0]
            return str(feedback) if isinstance(feedback, str) or not np.isnan(feedback) else ""
    else:
        return ""  # Eventually, Connect to string methods to get ta_feedback on an item if exists


def calculate_grade_difference(ai_grade, ta_grade):
    # Simple comparison of the ai_grade and ta_grade column. If the ta_grade is different, then the differential is positive. If the grade is different, then the differential is negative. No extra work needed.
    return ta_grade - ai_grade
