import os
from dotenv import load_dotenv
from groq import Groq
import typing_extensions as typing
import logging
import json
import streamlit as st

#GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def groq_model_generation(prompt: str, system_message: str, model: str) -> dict:
    """Model names: llama3_1, mixtral, gemma"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{system_message}"
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = response.choices[0].message.content
        logger.info(f"Response: {result}")

        # Parse result and raise exception if it's not valid JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error("Invalid JSON output string")
            raise

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def groq_inference(system_message: str, model_name: str, rules_list: list[str], sales_deck: str) -> typing.Optional[str]:
    """Perform inference using the groq api models and return the generated response."""
    output_list = []
    for rule in rules_list:
        input_text = f"""
        The rule is: {rule}
        The sales deck to evaluate is: {sales_deck}
        Your MUST provide an output in JSON representation with the following fields:
        "rule_name",
        "label",
        "part",
        "suggestion"
        """
        model_output = groq_model_generation(input_text, system_message, model_name)
        output_list.append(model_output)
        
    return output_list


def video_card_generation(transcript: str, model: str) -> str:
    """Model names: llama3_1, mixtral, gemma"""
    system_message = """
    Your task is to generate a concise summary from the given video transcript.
    Please follow these instructions return a markdown text:

    1. Extract Key Information:
    - Identify the company name.
    - Determine the industry, if applicable.
    - Summarize the product or service being discussed.

    2. Output Format:
    - **Company Name**: [Extracted company name]
    - **Industry**: [Extracted industry, if available]
    - **Product Summary**: [Brief summary of the product or service]

    Example:
    For a video discussing "FinGuard’s new portfolio management tool designed to help investors track and optimize their asset allocations," your output might look like:

    - Company Name: FinGuard
    - Industry: Financial Services
    - Product Summary: A portfolio management tool that assists investors in tracking and optimizing their asset allocations for improved investment outcomes.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{system_message}"
                },
                {
                    "role": "user",
                    "content": f"Here is the transcript to use: {transcript}",
                }
            ],
            model=model,
            temperature=0,
        )

        result = response.choices[0].message.content
        logger.info(f"Response: {result}")
        return result
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
