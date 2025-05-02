import re
from ta_feedback_analysis.select_utils import *
from ta_feedback_analysis.embeddings import add_embeddings
from tqdm import tqdm


FAULTY_TA_FEEDBACK = ['General', 'Your response has been saved', 'No submission',
                      'No Feedback was given by your instructor']


class AiTaFeedbackJoiner:
    def __init__(self, ai_feedback_path=AI_FEEDBACK_DATA_PATH, ta_feedback_path=TA_OVERRIDES_DATA_PATH,
                 joined_feedback_path=JOINED_FEEDBACK_PATH, feedback_differential_path=FEEDBACK_PATH,
                 selected_courses=None, selected_assignments=None, selected_tasks=None, selected_parts=None,
                 build_new=False):

        self.ai_feedback_path = ai_feedback_path
        self.ta_feedback_path = ta_feedback_path
        self.joined_feedback_path = joined_feedback_path
        self.feedback_differential_path = feedback_differential_path

        self.selected_courses = selected_courses
        self.selected_assignments = selected_assignments
        self.selected_tasks = selected_tasks
        self.selected_parts = selected_parts

        self.build_new = build_new
        self.fieldnames = ['course_id', 'course_name', 'assignment_id', 'assignment_name', 'task_id', 'student_id', 'task_title', 'grade_difference', 'ai_feedback',
                           'ta_feedback', 'feedback_additive_differential', 'feedback_additive_embedding',
                           'ta_id', 'ta_name']

    def join_ai_feedback_and_ta_feedback(self):
        ai_feedback = pd.read_csv(self.ai_feedback_path)
        ai_feedback = select_questions_to_process(ai_feedback, self.selected_courses, self.selected_assignments,
                                                  self.selected_tasks, self.selected_parts)

        ta_feedback = pd.read_csv(self.ta_feedback_path)
        ta_feedback = select_questions_to_process(ta_feedback, self.selected_courses, self.selected_assignments,
                                                  self.selected_tasks, self.selected_parts)

        ai_feedback.rename(columns={'user_id': 'student_id', 'grade': 'ai_grade'}, inplace=True)
        ta_feedback.rename(columns={'grade': 'ta_grade'}, inplace=True)

        joined_feedback = pd.merge(ai_feedback, ta_feedback, on=['course_id', 'assignment_id', 'task_id', 'student_id'],
                                   how='outer')
        joined_feedback.drop(columns=['course_name_y'], inplace=True)
        joined_feedback.rename(columns={'course_name_x': 'course_name'}, inplace=True)

        if not self.build_new and os.path.exists(self.joined_feedback_path):
            existing_joined_feedback = pd.read_csv(self.joined_feedback_path)
            joined_feedback = pd.concat([existing_joined_feedback, joined_feedback], ignore_index=True).drop_duplicates(
                subset=['course_id', 'assignment_id', 'task_id', 'student_id'])

        joined_feedback.to_csv(self.joined_feedback_path, index=False)

        return joined_feedback

    def build_feedback_differential_table(self):
        joined_feedback_df = pd.read_csv(JOINED_FEEDBACK_PATH)
        joined_feedback_df = select_questions_to_process(joined_feedback_df, self.selected_courses,  self.selected_assignments, self.selected_tasks, self.selected_parts)
        # diff_data = pd.DataFrame(columns=self.fieldnames)

        existing_data_exists = os.path.exists(self.feedback_differential_path) and not self.build_new

        if existing_data_exists:
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True

        for task_id in tqdm(joined_feedback_df['task_id'].unique()):
            filtered_df = joined_feedback_df[joined_feedback_df['task_id'] == task_id]

            for (student_id), student_response_group in filtered_df.groupby(['student_id']):
                try:
                    ai_grade = get_ai_grade(student_response_group)
                    ta_grade = get_ta_grade(student_response_group)
                    grade_difference = calculate_grade_difference(ai_grade, ta_grade)

                    ta_feedback = get_ta_feedback(student_response_group)
                    if ta_feedback != "":
                        x=1
                    for faulty_feedback in FAULTY_TA_FEEDBACK:
                        ta_feedback = ta_feedback.replace(faulty_feedback, '')
                    ai_feedbacks = []
                    feedback_part_names = []

                    for (part_name), student_part_response_group in student_response_group.groupby(['part_name']):
                        feedback_part_names.append(part_name[0])

                        single_ai_feedback = get_single_ai_feedback_for_part(student_part_response_group, part_name[0])
                        ai_feedbacks.append(single_ai_feedback)


                    ai_feedback = ' '.join(ai_feedbacks)
                    feedback_additive_differential = calculate_additive_feedback_differential(ai_feedback, ta_feedback, feedback_part_names)

                    new_row = {'course_id': student_response_group['course_id'].iloc[0],
                               'course_name': student_response_group['course_name'].iloc[0],
                               'assignment_id': student_response_group['assignment_id'].iloc[0],
                               'assignment_name': student_response_group['assignment_name'].iloc[0],
                               'task_id': student_response_group['task_id'].iloc[0],
                               'task_title': student_response_group['task_title'].iloc[0],
                               'student_id': student_id[0],
                               'grade_difference': grade_difference,
                               'ai_feedback': ai_feedback,
                               'ta_feedback': ta_feedback,
                               'feedback_additive_differential': feedback_additive_differential,
                               'ta_id': student_response_group['ta_id'].iloc[0],
                               'ta_name': f"{student_response_group['ta_first_name'].iloc[0]} {student_response_group['ta_last_name'].iloc[0]}"}

                    new_row = add_embeddings_if_needed(self.feedback_differential_path, new_row, existing_data_exists)

                    pd.DataFrame([new_row]).to_csv(self.feedback_differential_path, mode=mode, header=header, index=False)
                    mode = 'a'
                    header = False

                    # new_row_df = pd.DataFrame([new_row])
                    # diff_data = pd.concat([diff_data, new_row_df], ignore_index=True)
                except Exception as e:
                    print(f"Error processing task {task_id} for student {student_id}: {e}")

        # if self.build_new or not os.path.exists(self.feedback_differential_path):  # Building new feedback differential table
        #     diff_data.to_csv(self.feedback_differential_path, index=False)
        # else:
        #     combined_data = pd.concat([existing_data, diff_data]).drop_duplicates()
        #     combined_data.to_csv(self.feedback_differential_path, index=False)


def add_embeddings_if_needed(existing_data_path, new_data_row, existing_data_exists):
    if existing_data_exists:
        existing_data = pd.read_csv(existing_data_path)
        existing_row = existing_data[
            (existing_data['course_id'] == new_data_row['course_id']) &
            (existing_data['assignment_id'] == new_data_row['assignment_id']) &
            (existing_data['task_id'] == new_data_row['task_id']) &
            (existing_data['student_id'] == new_data_row['student_id'])
            ]

        if not existing_row.empty:  # If a matching row is found, use its embeddings
            new_data_row['feedback_additive_embedding'] = existing_row['feedback_additive_embedding'].iloc[0]

            return new_data_row
    else:   # If a matching row is found, use its embeddings
        new_data_row['feedback_additive_embedding'] = add_embeddings(new_data_row['feedback_additive_differential'])
    return new_data_row


def calculate_additive_feedback_differential(ai_feedback, ta_feedback, feedback_part_names=None):
    if feedback_part_names is not None:
        for part_name in feedback_part_names:
            ta_feedback = ta_feedback.replace(part_name, '')

    ai_sentences = set(re.split(r'(?<!\d)[.!?](?=\s+|[A-Z]|$)|\n+', ai_feedback))
    ta_sentences = set(re.split(r'(?<!\d)[.!?](?=\s+|[A-Z]|$)|\n+', ta_feedback))
    # If no changes were made, then show zero differential
    if ai_sentences == ta_sentences:
        return ""

    if '' in ai_sentences:
        ai_sentences.remove('')

    # Check if there is zero overlapping sentences implying all feedback sentences originate from TA.

    if not ai_sentences.intersection(ta_sentences):
        return ta_feedback

    # Find additional sentences originating from TA feedback
    additive_sentences = [sentence for sentence in ta_sentences if sentence not in ai_feedback]
    additive_feedback_differential = ' '.join(additive_sentences)
    return additive_feedback_differential


# df = pd.read_csv(FEEDBACK_DIFFERENTIAL_PATH)
# df['course_name'] = df['course_id']
# df.to_csv(FEEDBACK_DIFFERENTIAL_PATH, index=False)

# df = pd.read_csv(JOINED_FEEDBACK_PATH)
# df['course_name'] = df['course_id']
# df.to_csv(JOINED_FEEDBACK_PATH, index=False)

if __name__ == "__main__":
    feedback_joiner = AiTaFeedbackJoiner()
    feedback_joiner.join_ai_feedback_and_ta_feedback()
    feedback_joiner.build_feedback_differential_table()