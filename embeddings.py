import numpy as np
import ast
import pandas as pd
from openai_utils.openai_request import *
from openai_utils.openai_config import OpenAiOptions


def calculate_embedding(feedback, options=OpenAiOptions(model='text-embedding-3-small')):
    if (isinstance(feedback, str) and feedback == '') or pd.isna(feedback):
        embedding = pd.NA
    else:
        try:
            embedding = generate_embeddings(feedback, options, dimensions=256)
        except Exception as e:
            print(f"An error occurred while generating embeddings: {e}")
            embedding = pd.NA
    return embedding


def get_processed_embeddings(task_df, embedding_column):
    filtered_task_df = task_df.dropna(subset=[embedding_column]).reset_index(drop=True)  # Drop rows with no embeddings. If this df is all NaN, then the df will be empty/None
    embedding_df = filtered_task_df[embedding_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    if not all(embedding_df.isnull()):
        embedding_array = np.stack(embedding_df.values)
    else:
        embedding_array = None
    return filtered_task_df, embedding_array


def generate_embeddings(text, options: OpenAiOptions, dimensions=256):
    """
    Generate embeddings from OpenAI API with the given text and options.
    :param text: OpenAI text to generate embeddings.
    :param options: OpenAiOptions
    :param dimensions: The number of dimensions to return.

    :return: The embedding from OpenAI API.
    """
    config = OpenAiConfig()
    client = OpenAI(api_key=config.api_key, organization=config.organization)

    text = text.replace("\n", " ")

    embedding = client.embeddings.create(input=[text], model=options.model, dimensions=dimensions).data[0].embedding

    # if dimensions is not None:
    #     return embedding[:dimensions]
    return embedding