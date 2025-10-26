# imports
import os
import gradio as gr
from fastapi import FastAPI
from dotenv import load_dotenv
from scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI


app = FastAPI() #Creating instance of FastAPI

@app.get("/smc/")
def home(user_input:str):
    print("userinput: " + user_input)
    response = fact_checker( user_input + " site:factcheck.org OR site:who.int")
    return {"message": response}


def fact_checker(user_message): 
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')

    # Check the key
    if not api_key:
        print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
    elif not api_key.startswith("sk-proj-"):
        print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
    elif api_key.strip() != api_key:
        print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
    else:
        print("API key found and looks good so far!")


    # %%
    # To give you a preview -- calling OpenAI with these messages is this easy. Any problems, head over to the Troubleshooting notebook.
    system_prompt = """
    You are a fact-checking assistant that analyzes the user prompt text and the site to fact-check
    and return yes or no as a answer with a short explanation and percentage-score of credibility.
    List the sources and links of fact sources as is.
    """
    # message = "Hello, GPT! This is my first ever message to you! Hi!"
    message = user_message

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":message }
    ]
    print(messages)


    # %%
    openai = OpenAI()

    response = openai.chat.completions.create(model="gpt-5-nano", messages=messages)
    final_response= response.choices[0].message.content
    return final_response
    


def main():
    fact_checker("Vaccines cause autism site:factcheck.org OR site:who.int")


if __name__ == "__main__":
    main()
