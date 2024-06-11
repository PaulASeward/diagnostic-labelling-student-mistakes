from openai import OpenAI
from openai_utils.openai_config import OpenAiConfig, OpenAiOptions
from openai_utils.prompt_building import build_category_hint_prompt
import os
import openai
from dotenv import load_dotenv


def load_openai_env():
    load_dotenv()
    openai.organization = os.environ["OPENAI_ORGANIZATION"]
    openai.api_key = os.environ["OPENAI_API_KEY"]


def call_openai(prompt, options: OpenAiOptions):
    """
    Call OpenAI API with the given prompt and options.
    :param prompt: OpenAI prompt array of system and user messages.
    :param options: OpenAiOptions
    :return: The response json from OpenAI API.
    """

    try:
        return generate_chat_response(prompt, options)

    except openai.BadRequestError as e:
        print(f"Initial Bad OpenAI API request: {e}")
        options = OpenAiOptions(model='gpt-3.5-turbo-1106')  # Allow a retry with a different model with more tokens.
        return generate_chat_response(prompt, options)

    except openai.RateLimitError as e:  # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        print(f"OpenAI API request exceeded rate limit: {e}")
        return generate_chat_response(prompt, options, retry=True)


def generate_chat_response(prompt, options: OpenAiOptions, retry=False):
    """
    Generate a chat response from OpenAI API with the given prompt and options.
    :param prompt: OpenAI prompt array of system and user messages.
    :param options: OpenAiOptions
    :param retry: If retry request, use a different API key.
    :return: The response json from OpenAI API.
    """
    config = OpenAiConfig()
    client = OpenAI(api_key=config.api_key if not retry else config.RETRY_API_KEY, organization=config.organization)

    completion = client.chat.completions.create(
        model=options.model,
        messages=prompt,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
    )

    # response = completion.choices[0].message
    response = completion

    if options.log_response:
        print('Response returned: ', response)

    return response






