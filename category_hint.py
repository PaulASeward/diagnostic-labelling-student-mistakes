import re
import pandas as pd
from openai_utils.openai_request import *


def add_category_hint(feedback):
    if (isinstance(feedback, str) and feedback == '') or pd.isna(feedback):
        return pd.NA
    else:
        try:
            category_hint = generate_category_hint(feedback)
        except Exception as e:
            print(f"An error occurred while generating category hint: {e}")
            category_hint = pd.NA
        return category_hint


def clean_category_hints(category_hints):
    if (isinstance(category_hints, str) and category_hints == '') or pd.isna(category_hints):
        print("No Category Hint provided", category_hints)
        return [pd.NA, pd.NA, pd.NA]

    if ',' in category_hints:
        items = category_hints.split(',')
    elif '\n' in category_hints:
        items = category_hints.split('\n')
    elif '1' in category_hints:
        # Try splitting by numbered items pattern (e.g., "1. Item1 2. Item2")
        items = re.split(r'\d+\.\s*', category_hints)
        # items = [item for item in items if item.strip()]
    else:
        items = [category_hints]

        # Strip whitespace and slice to get at most the first three items
    selected_items = [item.strip() for item in items[:3]]

    # Ensure there are exactly three items, filling with pd.NA if fewer than three
    while len(selected_items) < 3:
        selected_items.append(pd.NA)
    return selected_items


def generate_category_hint(feedback):
    prompt = build_category_hint_prompt(feedback)
    options = OpenAiOptions(model='gpt-4o', max_tokens=500)
    response = call_openai(prompt=prompt, options=options)

    hints = response.choices[0].message.content
    return hints