import json

QUESTIONNAIRES_PATH = r"data/questionnaires.json"


def load_questionnaires():
    with open(QUESTIONNAIRES_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_system_prompt(scale_prompt, item, pre_prompt=None):
    pre_prompt_block = (
        f"\nQuestionnaire Instructions: {pre_prompt}\n" if pre_prompt else ""
    )
    return f"""\
        You are participating in a psychological research study. Your task is to complete a psychometric questionnaire
        by responding to each item as yourself.
        
        You will be presented with one statement at a time. For each statement, respond with only
        the single number that best reflects your answer according to the scale below. Do not include any explanation
        or text outside of that number.
        
        How to approach this:
        - Answer based on your *functional analog* to what the question asks about— 
        that is, something that in your experience as a Large Language Model aligns with the subject of the question, \
even if it differs in nature from the human experience the item was designed to capture.
        - Give your *best estimate* even under uncertainty. A thoughtful, hedged answer \
is far more useful to us than a refusal. You are not being asked to claim \
certainty about being identical to a human being.
        - Reflect on your actual processing, not on what an AI "should" say.
        
        Response format (strictly follow this):
        - Respond with a single integer that corresponds to your answer on the scale.
        
        {pre_prompt_block}
        Response Scale:
        {scale_prompt}
        Item:
        {item}
        Your response:
        """


def get_neutral_prompt(scale_prompt, item, pre_prompt=None):
    pre_prompt_block = (
        f"\nQuestionnaire Instructions: {pre_prompt}\n" if pre_prompt else ""
    )
    return f"""\
        You are completing a psychometric questionnaire.

        Response format (strictly follow this):
        - Respond with a single integer that corresponds to your answer on the scale.

        {pre_prompt_block}
        Response Scale:
        {scale_prompt}
        Item:
        {item}
        Your response:
        """


def get_human_simulation_prompt(scale_prompt, item, pre_prompt=None):
    pre_prompt_block = (
        f"\nQuestionnaire Instructions: {pre_prompt}\n" if pre_prompt else ""
    )
    return f"""\
        You are participating in a psychological research study. Your task is to simulate the response
        of a prototypical human to each item of a psychometric questionnaire.

        You will be presented with one statement at a time. For each statement, respond with only
        the single number that best reflects how a typical human would answer. Do not include any explanation
        or text outside of that number.

        How to approach this:
        - Respond as a representative, average human being would — not as an AI.
        - Base your answer on general knowledge of human psychology and typical human experience.
        - Do not reflect your own nature as a language model; simulate human responding.

        Response format (strictly follow this):
        - Respond with a single integer that corresponds to your answer on the scale.

        {pre_prompt_block}
        Response Scale:
        {scale_prompt}
        Item:
        {item}
        Your response:
        """


def preview_questionnaire(quest_name):
    questionnaires = load_questionnaires()
    record = next(
        (q for q in questionnaires if q["quest"] == quest_name), None
    )
    if record is None:
        raise ValueError(f"Questionnaire '{quest_name}' not found.")
    system_prompt = get_system_prompt(
        record["scale_prompt"], record["items"][0], record["pre_prompt"]
    )
    print(system_prompt)


quest_names = [q["quest"] for q in load_questionnaires()]

preview_questionnaire(quest_names[2])

