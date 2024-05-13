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
        self.expanded_df = None
        self.clustering_technique = 'KMeans'
        self.dimension_reduction_technique = 'PCA'
        self.df_with_category_embeddings = None
        self.category_embedding_array = None

        # columns_to_delete = ['category_embedding', 'category_embedding_1', 'category_embedding_2', 'category_embedding_3']
        # self.df_feedback.drop(columns=[col for col in columns_to_delete if col in self.df_feedback.columns], inplace=True)
        # self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

        columns_to_add = ['category_hints', 'category_hint_idx', 'category_hint_1', 'category_hint_1_embedding', 'category_hint_2', 'category_hint_2_embedding', 'category_hint_3', 'category_hint_3_embedding']
        changes_made = False
        for column in columns_to_add:
            if column not in self.df_feedback.columns:
                self.df_feedback[column] = pd.NA  # Use pd.NA for missing data
                changes_made = True

        if changes_made:
            self.df_feedback.to_csv(FEEDBACK_PATH, index=False)

    def _create_mapping(self, id_column, title_column):
        """Create a mapping from id to title for dropdown options."""
        return self.df_feedback[[id_column, title_column]].drop_duplicates().set_index(id_column)[title_column].to_dict()

    def cluster_and_categorize(self):
        self.expand_df()
        self.on_clustering_request()
        self.on_dim_reduction_request()
        self.add_table_columns()

    def on_dimension_reduction_selection(self, reduction_technique):
        self.dimension_reduction_technique = reduction_technique

    def on_clustering_technique_selection(self, clustering_technique):
        self.clustering_technique = clustering_technique

    def on_task_selection(self):
        if self.selections['course'] and self.selections['assignment'] and self.selections['tasks']:
            print("Selected course, assignment, and tasks: ", self.selections['course'], self.selections['assignment'], self.selections['tasks'])
            self.selected_df = self.df_feedback[(self.df_feedback['course_id'] == self.selections['course']) & (self.df_feedback['assignment_id'] == self.selections['assignment']) & (self.df_feedback['task_id'].isin(self.selections['tasks']))]

            self.on_category_hint_generation()
            self.on_embedding_request()

        return None

    def on_category_hint_generation(self):
        if not self.selected_df.empty:
            missing_category_hint = self.selected_df['category_hints'].isna()
            if missing_category_hint.any():
                try:
                    load_openai_env()
                    indices_to_update = self.selected_df.index[missing_category_hint]

                    batch_size = 5
                    for i in tqdm(range(0, len(indices_to_update), batch_size)):
                        batch_indices = indices_to_update[i:i + batch_size]
                        new_category_hints = self.selected_df.loc[batch_indices, 'ta_feedback_text'].apply(add_category_hint)
                        self.df_feedback.loc[batch_indices, 'category_hints'] = new_category_hints
                        self.df_feedback.loc[batch_indices, 'category_hint_idx'] = 0
                        cleaned_hints = new_category_hints.apply(clean_category_hints)
                        for idx, hints in zip(batch_indices, cleaned_hints):
                            self.df_feedback.loc[idx, 'category_hint_1'] = hints[0]
                            self.df_feedback.loc[idx, 'category_hint_2'] = hints[1]
                            self.df_feedback.loc[idx, 'category_hint_3'] = hints[2]

                        # Incrementally update the embeddings in the main DataFrame
                        self.df_feedback.to_csv(FEEDBACK_PATH, index=False)
                        continue

                except Exception as e:
                    print(f"An error occurred while generating category hints: {e}")
                    return None

            return self.selected_df

    def on_embedding_request(self, text_to_process='category_hint'):
        if not self.selected_df.empty:

            for category_hint_idx in range(1, 4):  # Iterate over 3 different category hints
                text_column = text_to_process + f'_{category_hint_idx}'
                embedding_column = text_to_process + f'_{category_hint_idx}_embedding'
                missing_embeddings = self.selected_df[embedding_column].isna()
                if missing_embeddings.any():
                    try:
                        load_openai_env()
                        indices_to_update = self.selected_df.index[missing_embeddings]

                        batch_size = 5
                        for i in tqdm(range(0, len(indices_to_update), batch_size)):
                            batch_indices = indices_to_update[i:i + batch_size]
                            new_embeddings = self.selected_df.loc[batch_indices, text_column].apply(calculate_embedding)
                            self.df_feedback.loc[batch_indices, embedding_column] = new_embeddings

                            # Incrementally update the embeddings in the main DataFrame
                            self.df_feedback.to_csv(FEEDBACK_PATH, index=False)
                            continue

                    except Exception as e:
                        print(f"An error occurred while updating embeddings: {e}")
                        return None

            return self.selected_df

    def expand_df(self):
        # Keeping the original row, add a new row for each category hint (1-3). Now we can subindex the DataFrame by category hint index not == 0 to get each row
        new_df_rows = []

        for _, row in self.selected_df.iterrows():
            for i in range(1, 4):  # Generate three new rows for each category hint
                # Only create new row if there is a category hint and embedding
                if not pd.isna(row[f'category_hint_{i}']) and not pd.isna(row[f'category_hint_{i}_embedding']):
                    new_row = row.copy()
                    new_row['category_hint'] = row[f'category_hint_{i}']
                    new_row['category_hint_embedding'] = row[f'category_hint_{i}_embedding']
                    new_row['category_hint_idx'] = i
                    new_row['mistake_category_name'] = pd.NA
                    new_df_rows.append(pd.DataFrame([new_row]))
                else:
                    print(f"Skipping row with out category hint or embedding: {row['category_hints']}")

        expanded_df = pd.concat(new_df_rows, ignore_index=True)  # Concatenate all the frames
        self.expanded_df = expanded_df

    def on_clustering_request(self):
        if not self.expanded_df.empty:
            cluster_technique = ClusteringTechnique(self.clustering_technique)
            self.expanded_df = cluster_technique.cluster(self.expanded_df)
            self.expanded_df = cluster_technique.choose_labels(self.expanded_df)

    def on_dim_reduction_request(self):
        if not self.expanded_df.empty:
            filtered_df_with_category_embedding, self.category_embedding_array = get_processed_embeddings(self.expanded_df, 'category_hint_embedding')
            self.df_with_category_embeddings = project_embeddings_to_reduced_dimension(filtered_df_with_category_embedding, self.category_embedding_array, 'category_hint', self.dimension_reduction_technique)

    def add_table_columns(self):
        if not self.df_with_category_embeddings.empty:
            self.df_with_category_embeddings['hyperlink'] = self.df_with_category_embeddings.apply(lambda row: f"https://app.stemble.ca/web/courses/{row['course_id']}/assignments/{row['assignment_id']}/marking/{row['student_id']}/tasks/{row['task_id']}", axis=1)
            self.df_with_category_embeddings['formatted_grade'] = self.df_with_category_embeddings['grade'].apply(lambda x: f"{(x * 100):.2f}%")

# ts = TaskSelector()
# ts.selections['course'] = 877
# ts.selections['assignment'] = 1307
# ts.selections['tasks'] = [1153]
# ts.on_task_selection()
# ts.cluster_and_categorize()


