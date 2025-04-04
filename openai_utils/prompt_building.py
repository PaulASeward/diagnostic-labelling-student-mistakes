CATEGORY_HINT_PROMPT = (
    "Given this feedback from a student, return a comma separated list of descriptive titles (1-10 word max) to summarize the student's feedback for future analytical purposes. List MUST BE OF MAX 3 items but may be less. Return the string, 'No feedback' if student's feedback was not relevant."
)
# CATEGORY_HINT_PROMPT = (
#     "Given this feedback given to a student, return a comma separated list of diagnostic labels (1-6 word max) to categorize the student's mistakes for future analytical purposes, with the most important mistakes first. List MUST BE OF MAX 3 items but may be less. Be as specific as possible with what the student got wrong. Return the string, 'No mistakes' if student made no significant mistakes."
# )


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
