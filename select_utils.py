import numpy as np
import pandas as pd
from embeddings import *
from tqdm import tqdm
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
        # self.df_feedback['feedback_embedding'] = pd.NA
        # self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

        # CHeck if feedback_embedding column exists:
        if 'feedback_embedding' not in self.df_feedback.columns:
            self.df_feedback['feedback_embedding'] = pd.NA
            self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

        if 'category_hint' not in self.df_feedback.columns:
            self.df_feedback['category_hint'] = pd.NA
            self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

        if 'category_embedding' not in self.df_feedback.columns:
            self.df_feedback['category_embedding'] = pd.NA
            self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

    def _create_mapping(self, id_column, title_column):
        """Create a mapping from id to title for dropdown options."""
        return self.df_feedback[[id_column, title_column]].drop_duplicates().set_index(id_column)[title_column].to_dict()

    def on_task_selection(self):
        if self.selections['course'] and self.selections['assignment'] and self.selections['tasks']:
            self.selected_df = self.df_feedback[(self.df_feedback['course_id'] == self.selections['course']) &
                                            (self.df_feedback['assignment_id'] == self.selections['assignment']) &
                                            (self.df_feedback['task_id'].isin(self.selections['tasks']))]

            return self.selected_df
        return None

    def on_feedback_embedding_request(self):
        if not self.selected_df.empty:
            missing_embeddings = self.selected_df['feedback_embedding'].isna()
            if missing_embeddings.any():
                try:
                    load_openai_env()
                    indices_to_update = self.selected_df.index[missing_embeddings]

                    batch_size = 3
                    for i in tqdm(range(0, len(indices_to_update), batch_size)):
                        batch_indices = indices_to_update[i:i + batch_size]
                        new_embeddings = self.selected_df.loc[batch_indices, 'ta_feedback_text'].apply(add_embeddings)
                        self.df_feedback.loc[batch_indices, 'feedback_embedding'] = new_embeddings

                        # Incrementally update the embeddings in the main DataFrame
                        self.df_feedback.to_csv(FEEDBACK_PATH, index=False)
                        continue

                except Exception as e:
                    print(f"An error occurred while updating embeddings: {e}")
                    return None

            return self.selected_df

    def on_category_hint_generation(self):
        if not self.selected_df.empty:
            missing_category_hint = self.selected_df['category_hint'].isna()
            if missing_category_hint.any():
                load_openai_env()
                self.df_feedback.loc[self.selected_df.index[missing_category_hint], 'category_hint'] = self.selected_df.loc[missing_category_hint, 'ta_feedback_text'].apply(add_category_hint)

                # Save the updated DataFrame
                self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

            return self.selected_df




ts = TaskSelector()
ts.selections['course'] = 877
ts.selections['assignment'] = 1302
ts.selections['tasks'] = [691]


ts.on_task_selection()
# ts.on_feedback_embedding_request()
ts.on_category_hint_generation()