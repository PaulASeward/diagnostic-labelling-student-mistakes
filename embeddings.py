import numpy as np
import ast
import re
import pandas as pd
from openai_utils.openai_request import *
from openai_utils.openai_config import OpenAiOptions


def calculate_embedding(feedback, options=OpenAiOptions(model='text-embedding-ada-002', max_tokens=300)):
    if (isinstance(feedback, str) and feedback == '') or pd.isna(feedback):
        embedding = pd.NA
    else:
        try:
            embedding = generate_embeddings(feedback, options, dimensions=600)
        except Exception as e:
            print(f"An error occurred while generating embeddings: {e}")
            embedding = pd.NA
    return embedding


def add_category_hint(feedback):
    if (isinstance(feedback, str) and feedback == '') or pd.isna(feedback):
        return pd.NA
    else:
        try:
            category_hint = generate_category_hint(feedback)
            print(f"Category hint: {category_hint}")
        except Exception as e:
            print(f"An error occurred while generating category hint: {e}")
            category_hint = pd.NA
        return category_hint


def clean_category_hints(category_hints):
    if (isinstance(category_hints, str) and category_hints == '') or pd.isna(category_hints):
        return [pd.NA, pd.NA, pd.NA]
    else:
        if ',' in category_hints:
            items = category_hints.split(',')
        elif '\n' in category_hints:
            items = category_hints.split('\n')
        elif '1' in category_hints:
            # Try splitting by numbered items pattern (e.g., "1. Item1 2. Item2")
            items = re.split(r'\d+\.\s*', category_hints)
            items = [item for item in items if item.strip()]
        else:
            print("No Splitting Separator found in: ", category_hints)
            items = [category_hints]

            # Strip whitespace and slice to get at most the first three items
        selected_items = [item.strip() if item.strip() != 'None' or item.strip() != 'N/A' else pd.NA for item in items[:3]]

        # Ensure there are exactly three items, filling with pd.NA if fewer than three
        while len(selected_items) < 3:
            selected_items.append(pd.NA)
        return selected_items


def get_processed_embeddings(task_df, embedding_column):
    filtered_task_df = task_df.dropna(subset=[embedding_column]).reset_index(drop=True)  # Drop rows with no embeddings. If this df is all NaN, then the df will be empty/None
    embedding_df = filtered_task_df[embedding_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    if not all(embedding_df.isnull()):
        embedding_array = np.stack(embedding_df.values)
    else:
        embedding_array = None
    return filtered_task_df, embedding_array
