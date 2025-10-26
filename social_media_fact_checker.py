# imports
import os
import gradio as gr
from fastapi import FastAPI
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

#Load necessary variables
load_dotenv(override=True) # Loads variables from .env file
api_key = os.getenv('OPENAI_API_KEY') #Gets the Open API key 


def fact_checker(user_message, selected_category): #Main function that conducts fact-checking
    
    # Check the key and return corresponding message 
    if not api_key:
        return "❌ Error: No API key was found. Please check your .env file." #If no API key was found
    elif not api_key.startswith("sk-proj-"):
        return "❌ Error: API key doesn't start with sk-proj-. Please check you're using the right key."#If API key doesn't start with espected prefix
    elif api_key.strip() != api_key:
        #If API key has leading, trailing spaces/tabs
        return "❌ Error: An API key was found, but it looks like it might have space or tab characters at the start or end "
    
    print("API key found and looks good!") #Testing message
    
    #mapping out dropdown selection to site constraints
    category_sites = {
        "general": ["site:reuters.com", "site:apnews.com"],
        "us_politics": ["site:factcheck.org", "site:politifact.com"],
        "us_legislation": ["site:congress.gov", "site:govtrack.us"],
        "economy_labor": ["site:bls.gov", "site:fred.stlouisfed.org"],
        "public_health": ["site:cdc.gov", "site:who.int"],
        "medicine": ["site:pubmed.ncbi.nlm.nih.gov", "site:medlineplus.gov"],
        "research": ["site:arxiv.org", "site:nature.com"],

    }
    if selected_category in category_sites:
        selected_sites = category_sites[selected_category]
        site_filter_text = " AND/OR ".join(selected_sites)
        system_prompt = f"""
        You are a fact-checking assistant that analyzes the user prompt text and checks its validity.
        Return 'yes' or 'no' with a short explanation and a percentage credibility score.
        Only use the following two sites: {site_filter_text}
        List the sources and links as they are.
        Credibility score should reflect percentage accuracy of user-inputs. """

    else:
         system_prompt = """
        You are a fact-checking assistant that analyzes the user prompt text and checks its validity.
        Return 'yes' or 'no' with a short explanation and a percentage credibility score.
        You may use any credible source.
        List the sources and links as they are.
        """ #Simple prompt to tell model how to behave when user does NOT want constrained results

    messages = [ # Essentially create a conversation structure for user-input vs. model 
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try: #Attempt to execute code, catch + print errors if they occur
        openai = OpenAI() #Creates OpenAI instance/client--> Uses OPENAI_API_KEY from environment
        response = openai.chat.completions.create(
            model="gpt-4-turbo",  
            messages=messages #Send request/message to OpenAI() instance 
        )
        fact_check_result = response.choices[0].message.content.strip()

        is_false = fact_check_result.lower().startswith("no") or (
            "credibility score: " in fact_check_result.lower() and "50%" in fact_check_result.lower()
        )

        if is_false:
            counter_prompt = f"""
            The following claim has been determined to be false:
            "{user_message}"
            
            Please write a concise, factual, and respectful counter-response that can be used to correct misinformation in a discussion.
            Make sure it's under 280 characters and easy to copy and paste.
            """
            counter_response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You generate counter-arguments for false claims."},
                    {"role": "user", "content": counter_prompt}
                ]
            )
            counter_text = counter_response.choices[0].message.content.strip()

            return f"🧠 **Fact Check Result:**\n{fact_check_result}", counter_text
        else:
            return f"🧠 **Fact Check Result:**\n{fact_check_result}", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", "", True #Return error message, empty counter-argument, and True to indicate error

def check_claim(claim, selected_category):
    if not claim.strip():
        return "⚠️ Please enter a claim to fact-check.", ""
    
    print(f"Checking claim: {claim} | Category: {selected_category}")
    result, counter = fact_checker(claim, selected_category)
    print(f"Fact-check result: {result}")
    return result, counter


with gr.Blocks(title="Fact Checker AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Fact Checker AI
        ### Verify claims and statements with AI-powered fact-checking
        """
    )

    with gr.Row():
        # Left side: Input section (1/3 width)
        with gr.Column(scale=1):
             claim_input = gr.Textbox(
                 label="Enter Claim to Fact-Check",
                 placeholder="Example: Vaccines cause autism",
                 lines=3
             )

             category_dropdown = gr.Dropdown(
                 label="Select Category",
                 choices=[
                     "General Claim",
                     "U.S. Politics",
                     "U.S. Legislation",
                     "Economy & Labor",
                     "Public Health",
                     "Medicine",
                     "Research"
                 ],
                 value="General Claim"
             )

             # Examples directly under claim input
             gr.Examples(
                 examples=[
                     ["Vaccines cause autism", "Medicine"],
                     ["The Earth is flat", "General Claim"],
                     ["Coffee is bad for your health", "Public Health"],
                     ["5G networks spread COVID-19", "Public Health"],
                 ],
                 inputs=[claim_input, category_dropdown],
                 label="🧪 Try an Example"
             )

             with gr.Row():
                 submit_btn = gr.Button("🔍 Check Fact", variant="primary", scale=2)
                 clear_btn = gr.Button("🗑️ Clear", scale=1)

        # Right side: Output section (2/3 width)
        with gr.Column(scale=2):
            output = gr.Markdown(
                label="Fact-Check Result",
                value="Results will appear here..."
            )

            counter_box = gr.Textbox(
                label="Suggested Counter-Argument",
                lines=3,
                visible=True,
                value=""
            )

    # Text fact-check logic
    def check_and_display(claim, selected_category):
        result_text, counter_argument = check_claim(claim, selected_category)
        return (
            result_text,
            gr.update(value=counter_argument, visible=True)
        )

    # Text events
    submit_btn.click(fn=check_claim, inputs=[claim_input, category_dropdown], outputs=[output, counter_box])
    clear_btn.click(fn=lambda: ("", "Results will appear here..."), outputs=[claim_input, output])
    claim_input.submit(fn=check_claim, inputs=[claim_input, category_dropdown], outputs=[output, counter_box])


def main():
    # Launch Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  
    )

if __name__ == "__main__":
    main()