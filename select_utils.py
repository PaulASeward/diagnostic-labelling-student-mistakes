import numpy as np
import pandas as pd
from embeddings import get_processed_embeddings, add_embeddings
from dimension_reduction import project_embeddings_to_reduced_dimension

FEEDBACK_PATH = 'data/feedback.csv'


class TaskSelector:
    def __init__(self, feedback_path=FEEDBACK_PATH):
        self.feedback_path = feedback_path
        self.df_feedback = pd.read_csv(feedback_path)  # Load the larger DataFrame for mappings
        self.course_mapping = self._create_mapping('course_id', 'course_name')
        self.assignment_mapping = self._create_mapping('assignment_id', 'assignment_name')
        self.task_mapping = self._create_mapping('task_id', 'task_title')
        self.selections = {'course': None, 'assignment': None, 'tasks': []}
        self.selected_df = None

        # CHeck if feedback_embedding column exists:
        if 'feedback_embedding' not in self.df_feedback.columns:
            self.df_feedback['feedback_embedding'] = pd.NA
            self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

    def _create_mapping(self, id_column, title_column):
        """Create a mapping from id to title for dropdown options."""
        return self.df_feedback[[id_column, title_column]].drop_duplicates().set_index(id_column)[title_column].to_dict()

    def on_task_selection(self):
        selected_course = self.selections['course']
        selected_assignment = self.selections['assignment']
        selected_tasks = self.selections['tasks']

        if selected_course and selected_assignment and selected_tasks:
            # # Obtain indices of selected tasks
            # selected_indices = np.indices(self.df_feedback[(self.df_feedback['course_id'] == selected_course) &
            #                                 (self.df_feedback['assignment_id'] == selected_assignment) &
            #                                 (self.df_feedback['task_id'].isin(selected_tasks))])
            #
            # # Calculate embeddings on rows without feedback embedding:
            # missing_embedding_rows = np.indices(self.df_feedback[selected_indices]['feedback_embedding'].isna())
            #
            # if missing_embedding_rows.size != 0:
            #     self.df_feedback.loc[missing_embedding_rows, 'feedback_embedding'] = self.df_feedback.loc[missing_embedding_rows, 'feedback'].apply(add_embeddings)
            #
            #     # Save the updated DataFrame
            #     self.df_feedback.to_csv(FEEDBACK_PATH, index=False)
            #
            # self.selected_df = self.df_feedback[selected_indices]

            selected_df = self.df_feedback[(self.df_feedback['course_id'] == selected_course) &
                                            (self.df_feedback['assignment_id'] == selected_assignment) &
                                            (self.df_feedback['task_id'].isin(selected_tasks))]

            missing_embeddings = selected_df['feedback_embedding'].isna()
            if missing_embeddings.any():
                self.df_feedback.loc[selected_df.index[missing_embeddings], 'feedback_embedding'] = selected_df.loc[missing_embeddings, 'feedback'].apply(add_embeddings)

                # Save the updated DataFrame
                self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

            self.selected_df = selected_df


ts = TaskSelector()
ts.selections['course'] = 877
ts.selections['assignment'] = 1302
ts.selections['tasks'] = [691]


ts.on_task_selection()