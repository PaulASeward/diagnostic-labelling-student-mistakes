CATEGORY_HINT_PROMPT = (
    "Given this feedback given to a student, return a comma separated list of diagnostic labels (1-4 word max) to categorize the student's mistakes for future analytical purposes, with the most important mistakes first. List MUST BE OF MAX 3, and can be only 1 or 2 items. Return the string, 'None' if student made no actual mistakes."
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
