import numpy as np
import ast

import pandas as pd
from openai_utils.openai_request import *
from openai_utils.openai_config import OpenAiOptions


def produce_embeddings_for_task(task_csv_path):
    task_df = pd.read_csv(task_csv_path)
    task_df['feedback_additive_embedding'] = task_df['feedback_additive_differential'].apply(add_embeddings)
    task_df['feedback_subtractive_embedding'] = task_df['feedback_subtractive_differential'].apply(add_embeddings)

    task_df.to_csv(task_csv_path, index=False)
    return task_df


def calculate_embedding(feedback, options=OpenAiOptions(model='text-embedding-ada-002', max_tokens=300)):
    embedding = generate_embeddings(feedback, options, dimensions=None)
    return embedding


def add_embeddings(feedback_diff):
    if (isinstance(feedback_diff, str) and feedback_diff == '') or pd.isna(feedback_diff):
        return pd.NA
    else:
        return calculate_embedding(feedback_diff)

def get_processed_embeddings(task_df, diff_type_prefix):
    embedding_column = f'feedback_{diff_type_prefix}_embedding'
    filtered_task_df = task_df.dropna(subset=[embedding_column]).reset_index(drop=True)  # Drop rows with no embeddings. If this df is all NaN, then the df will be empty/None
    embedding_df = filtered_task_df[embedding_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    if not all(embedding_df.isnull()):
        embedding_array = np.stack(embedding_df.values)
    else:
        embedding_array = None
    return filtered_task_df, embedding_array