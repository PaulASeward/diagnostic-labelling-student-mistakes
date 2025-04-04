from category_hint import *
from tqdm import tqdm
import json
import struct
from select_utils import FEEDBACK_PATH


def transform_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # df['ta_feedback_text'] = pd.NA
    # df['category_hints'] = pd.NA
    # df['category_hint_1'] = pd.NA
    # df['category_hint_2'] = pd.NA
    # df['category_hint_3'] = pd.NA
    # df['category_hint_1_embedding'] = pd.NA
    # df['category_hint_2_embedding'] = pd.NA
    # df['category_hint_3_embedding'] = pd.NA
    # df['category_hint_idx'] = pd.NA



    valid_rows = ~df['category_hints'].str.contains(
        r'No Mistake|3\. No Mistakes|No Mistakes|No mistakes|No mistake|N/A|None',
        case=False, na=False
    )
    df = df[valid_rows]

    # Check if 'mistake_label' column exists
    uses_old_mistake_label_column = 'mistake_label' in df.columns
    has_feedback_data_column = 'feedback_data' in df.columns
    if has_feedback_data_column:
        df['ta_feedback_text'] = pd.NA

    # Iterate over the grouped DataFrame
    for (course_id, assignment_id, task_id, user_id), student_grades_group in tqdm(
            df.groupby(['course_id', 'assignment_id', 'task_id', 'user_id'])):
        for idx in student_grades_group.index:
            # Save the mistake label into category_hints and category_hint_1
            if uses_old_mistake_label_column:
                mistake_label = df.at[idx, 'mistake_label']
                df.at[idx, 'category_hints'] = mistake_label

            # Clean the category hints into category_hint_1, category_hint_2, and category_hint_3
            category_hints = df.at[idx, 'category_hints']
            if pd.isna(category_hints):
                continue

            cleaned_hints = clean_category_hints(category_hints)
            for i, hint in enumerate(cleaned_hints):
                df.at[idx, f'category_hint_{i + 1}'] = hint

            # Unpack binary string representation of the embedding and save to category_hint_1_embedding
            embedding = df.at[idx, 'category_hint_1_embedding']
            if pd.notna(embedding):
                try:
                    # Convert the hexadecimal string to bytes
                    if isinstance(embedding, str):
                        embedding_bytes = bytes.fromhex(embedding[2:])  # Skip "0x" prefix
                    else:
                        embedding_bytes = embedding

                    # Ensure the length matches the expected size (1024 bytes)
                    if len(embedding_bytes) == 1024:
                        embedding_list = struct.unpack(f'{256}f', embedding_bytes)
                        df.at[idx, 'category_hint_1_embedding'] = list(embedding_list)
                    else:
                        raise ValueError(f"Unexpected embedding size: {len(embedding_bytes)} bytes")
                except (struct.error, ValueError) as e:
                    print(f"Error processing embedding at index {idx}: {e}")

            # Extract TA feedback text from feedback_data JSON object
            if has_feedback_data_column:
                feedback_data = df.at[idx, 'feedback_data']
                if pd.notna(feedback_data):
                    try:
                        feedback_text = json.loads(feedback_data).get('feedback', None)
                        df.at[idx, 'ta_feedback_text'] = feedback_text
                    except json.JSONDecodeError:
                        pass

    df = df.drop(columns=['mistake_label', 'feedback_data', 'embedding'], errors='ignore')
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    transform_data(FEEDBACK_PATH, FEEDBACK_PATH)


