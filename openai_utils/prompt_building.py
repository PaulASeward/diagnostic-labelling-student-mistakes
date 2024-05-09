CATEGORY_HINT_PROMPT = (
    "Given this feedback given to a student, return a (1-4 word max) diagnostic label to categorize the student's mistake for future analytical purposes."
)


def build_category_hint_prompt(feedback):
    result = []

    prompt = CATEGORY_HINT_PROMPT

    result.append({
        "role": "system",
        "content": prompt,
    })

    result.append({
        "role": "user",
        "content": feedback,
    })

    return result
