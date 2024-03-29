import openai
import re
import json
import time
import tiktoken 
from utils.database import *
from .converter import pdf2string,txt2string
# from utils.logger_config import *
# from main import logger
import time 
from utils.logger_config import *
logger=setup_logger(__name__, "parser.log", level=logging.INFO)

OPENAI_settings=config_col.find_one({"config_name":"openai_settings"})
OPENAI_API_KEY=OPENAI_settings['api_key'] 
openai.api_key = OPENAI_API_KEY

# Prompts=config_col.find_one({"config_name":"prompts"})["prompts"]
# prompt_questions=Prompts["resume"] 
# prompt_questions2 =Prompts["jobads2"]
Prompts=config_col.find_one({"config_name":"prompts"})["prompts"]

query_openai_settings={"config_name":"openai_settings"}
openai_settings=config_col.find_one(query_openai_settings)
max_tokens = openai_settings['max_tokens']
engine = openai_settings['engine']
temperature=openai_settings['temperature']
top_p=openai_settings['top_p']
frequency_penalty=openai_settings['frequency_penalty']
presence_penalty=openai_settings['presence_penalty']
# Rest of your functions/methods with added logging statements

def get_maxtoken(text,model,max_token):
    try:
        encoding = tiktoken.encoding_for_model(model)
        logger.info('Successfully obtained max token')
        return max_token-1-len(encoding.encode(text))
    except Exception as e:
        logger.error('Error while obtaining max token: ', exc_info=True)
        raise

def get_token(text,model):
    try:
        encoding = tiktoken.encoding_for_model(model)
        logger.info('Successfully obtained token')
        return len(encoding.encode(text))
    except Exception as e:
        logger.error('Error while obtaining token: ', exc_info=True)
        raise

def get_message(text):
    try:
        content="""
        Using the following job description, please generate  multiple-choice questions that test the candidate's suitability for the role, it's not for testing their understanding of the key requirements and responsibilities. Each answer is scored from 1-5 with the best answer being 5, 1 being the worst answer. output JSON  the following structure: [{question , answers: [{answer, score, rationale}]}]:

         {text}
        """.replace("{text}",text)
        message= [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": content
            }
        ]
        logger.info('Message created successfully.')
        return message
    except Exception as e:
        logger.error('Error while creating message: ', exc_info=True)
        raise


def generate_questions(text):
    start = time.time()

    try:
        questions = []
        messages=get_message(text)
        for _ in range(0,3):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",  # Replace with the correct model
                        messages=messages,
                        max_tokens=4000,
                    )

                    # Assuming the OpenAI API returns the questions in the desired format
                    generated_questions = json.loads(response['choices'][0]['message']['content'].strip())
                    questions.extend(generated_questions)

                    # Add the last assistant's message to the conversation to build on the previous response
                    messages.append({
                        "role": "assistant",
                        "content": response['choices'][0]['message']['content']
                    })
                    if len(questions) >=8:
                        break
                except Exception as e:
                    logger.error('Error while generating questions: ', exc_info=True)

        end = time.time()
        logger.info(f"Time taken: {end - start} seconds")
        return questions
    except Exception as e:
        logger.error('Error in generate_questions function: ', exc_info=True)
        raise
