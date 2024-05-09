import numpy as np
import pandas as pd
from clustering_utils import *
from embeddings import *
from tqdm import tqdm
from dimension_reduction import project_embeddings_to_reduced_dimension

FEEDBACK_PATH = 'data/feedback.csv'


class TaskSelector:
    def __init__(self, feedback_path=FEEDBACK_PATH):
        self.feedback_path = feedback_path
        self.df_feedback = pd.read_csv(feedback_path)
        self.course_mapping = self._create_mapping('course_id', 'course_name')
        self.assignment_mapping = self._create_mapping('assignment_id', 'assignment_name')
        self.task_mapping = self._create_mapping('task_id', 'task_title')
        self.selections = {'course': None, 'assignment': None, 'tasks': []}
        self.selected_df = None
        self.dimension_reduction_technique = 'PCA'
        self.df_with_feedback_embedding = None
        self.df_with_category_embedding = None
        self.feedback_embedding_array = None
        self.category_embedding_array = None

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

            self.on_embedding_request('ta_feedback_text')
            self.on_embedding_request('category_hint')
            self.on_category_hint_generation()
            self.on_clustering_request()
            self.on_dim_reduction_request()

            return self.selected_df
        return None

    def on_clustering_request(self):
        if not self.selected_df.empty:
            self.selected_df = cluster_student_mistakes_kmeans(self.selected_df, embedding_type_prefix='feedback')
            self.selected_df = cluster_student_mistakes_kmeans(self.selected_df, embedding_type_prefix='category')

    def on_embedding_request(self, text_to_process):
        embedding_column = 'feedback_embedding' if text_to_process == 'ta_feedback_text' else 'category_embedding'

        if not self.selected_df.empty:
            missing_embeddings = self.selected_df[embedding_column].isna()
            if missing_embeddings.any():
                try:
                    load_openai_env()
                    indices_to_update = self.selected_df.index[missing_embeddings]

                    batch_size = 3
                    for i in tqdm(range(0, len(indices_to_update), batch_size)):
                        batch_indices = indices_to_update[i:i + batch_size]
                        new_embeddings = self.selected_df.loc[batch_indices, text_to_process].apply(calculate_embedding)
                        self.df_feedback.loc[batch_indices, embedding_column] = new_embeddings

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
                try:
                    load_openai_env()
                    indices_to_update = self.selected_df.index[missing_category_hint]

                    batch_size = 3
                    for i in tqdm(range(0, len(indices_to_update), batch_size)):
                        batch_indices = indices_to_update[i:i + batch_size]
                        new_category_hints = self.selected_df.loc[batch_indices, 'ta_feedback_text'].apply(add_category_hint)
                        self.df_feedback.loc[batch_indices, 'category_hint'] = new_category_hints

                        # Incrementally update the embeddings in the main DataFrame
                        self.df_feedback.to_csv(FEEDBACK_PATH, index=False)
                        continue

                except Exception as e:
                    print(f"An error occurred while generating category hints: {e}")
                    return None

            return self.selected_df

    def on_dim_reduction_request(self):
        if not self.selected_df.empty:
            filtered_df_with_feedback_embedding, self.feedback_embedding_array = get_processed_embeddings(self.selected_df, 'feedback_embedding')
            filtered_df_with_category_embedding, self.category_embedding_array = get_processed_embeddings(self.selected_df, 'category_embedding')

            self.df_with_feedback_embedding = project_embeddings_to_reduced_dimension(filtered_df_with_feedback_embedding, self.feedback_embedding_array, 'feedback', self.dimension_reduction_technique)
            self.df_with_category_embedding = project_embeddings_to_reduced_dimension(filtered_df_with_category_embedding, self.category_embedding_array, 'category', self.dimension_reduction_technique)


# ts = TaskSelector()
# ts.selections['course'] = 877
# ts.selections['assignment'] = 1302
# ts.selections['tasks'] = [691]
# ts.on_task_selection()

# ts.on_dim_reduction_request()
# ts.on_clustering_request()
# ts.on_category_hint_generation()
# # ts.on_embedding_request('ta_feedback_text')
# ts.on_embedding_request('category_hint')
