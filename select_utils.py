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
        self.clustered_df = None
        self.df_with_category_embeddings = None

        self.cluster_algorithm = ClusterAlgorithm('KMeans', n_clusters=5)
        self.dimension_reduction_technique = 'PCA'
        self.color_map = None
        self.number_mistake_labels = 3
        self.dendrogram_label = 'mistake_category_name'
        self.dendrogram_color = False

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

    def on_cluster_config_selection(self, clustering_technique='KMeans', n_clusters=5):
        self.cluster_algorithm.clustering_technique = clustering_technique if clustering_technique else 'KMeans'
        self.cluster_algorithm.n_clusters = n_clusters if n_clusters else 5

    def on_manual_categories_selection(self, mistake_table_current_data):
        manual_categories = [row['option'] for row in mistake_table_current_data]
        set_manual_categories = set(manual_categories)
        existing_categories = self.cluster_algorithm.mistake_categories_dict.keys()
        set_existing_categories = set(existing_categories)
        n_clusters = len(manual_categories)

        if manual_categories and set_manual_categories != set_existing_categories and n_clusters > 0 and self.cluster_algorithm.use_manual_mistake_categories > 0:
            self.cluster_algorithm.n_clusters = n_clusters
            new_mistake_categories_dict = {}

            for manual_category in manual_categories:
                if manual_category not in set_existing_categories:
                    new_mistake_categories_dict[manual_category] = calculate_embedding(manual_category)
                else:
                    new_mistake_categories_dict[manual_category] = self.cluster_algorithm.mistake_categories_dict[manual_category]  # Reuse calculated embedding

            self.cluster_algorithm.mistake_categories_dict = new_mistake_categories_dict

    def load_task(self):
        self.selected_df = self.df_feedback[(self.df_feedback['course_id'] == self.selections['course']) & (self.df_feedback['assignment_id'] == self.selections['assignment']) & (self.df_feedback['task_id'].isin(self.selections['tasks']))]

    def on_task_selection(self):
        if self.selections['course'] and self.selections['assignment'] and self.selections['tasks']:
            print("Selected course, assignment, and tasks: ", self.selections['course'], self.selections['assignment'], self.selections['tasks'])
            self.load_task()
            self.on_category_hint_generation()
            self.on_embedding_request()
            self.load_task()
            self.expand_df()
        return None

    def update_table(self, current_selection_indices, task_embeddings_df):
        filtered_df = task_embeddings_df[task_embeddings_df.apply(
            lambda row: (row['student_id'], row['category_hint_idx']) in current_selection_indices, axis=1)]
        filtered_df['formatted_grade'] = filtered_df.apply(lambda row: f"[{(row['grade'] * 100):.2f}%](https://app.stemble.ca/web/courses/{row['course_id']}/assignments/{row['assignment_id']}/marking/{row['student_id']}/tasks/{row['task_id']})", axis=1)
        filtered2_df = filtered_df[['mistake_category_name', 'category_hint', 'ta_feedback_text', 'category_hints', 'formatted_grade']]
        data_to_display = filtered2_df.to_dict('records')
        return data_to_display

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
            for i in range(1, self.number_mistake_labels + 1):  # Generate up to three new rows for each category hint depending on the number of mistakes allowed
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

        self.expanded_df = pd.concat(new_df_rows, ignore_index=True)  # Concatenate all the frames

    def on_clustering_request(self):
        if not self.expanded_df.empty and self.cluster_algorithm:
            # Only use the specified amount of mistake labels per student
            self.clustered_df = self.expanded_df[self.expanded_df['category_hint_idx'] <= self.number_mistake_labels]
            self.clustered_df = self.cluster_algorithm.cluster(self.clustered_df)
            self.clustered_df = self.cluster_algorithm.choose_labels(self.clustered_df)

    def on_dim_reduction_request(self):
        if not self.clustered_df.empty:
            filtered_df_with_category_embedding, category_embedding_array = get_processed_embeddings(self.clustered_df, 'category_hint_embedding')
            self.df_with_category_embeddings = project_embeddings_to_reduced_dimension(filtered_df_with_category_embedding, category_embedding_array, 'category_hint', self.dimension_reduction_technique)
            self.df_with_category_embeddings.sort_values(by='mistake_category_label', inplace=True)


# ts = TaskSelector()
# ts.selections['course'] = 879
# ts.selections['assignment'] = 1539
# ts.selections['tasks'] = [1128]
# ts.on_task_selection()
# ts.on_clustering_request()


