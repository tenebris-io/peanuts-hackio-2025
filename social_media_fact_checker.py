# imports
import os
import gradio as gr
from fastapi import FastAPI
from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI
from transformers import pipeline
import numpy as np

#Load necessary variables
load_dotenv(override=True) # Loads variables from .env file
api_key = os.getenv('OPENAI_API_KEY') #Gets the Open API key 

#Iniitlaize speech-to-text model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")



def fact_checker(user_message, constrained_results): #Main function that conducts fact-checking
    
    # Check the key and return corresponding message 
    if not api_key:
        return "❌ Error: No API key was found. Please check your .env file." #If no API key was found
    elif not api_key.startswith("sk-proj-"):
        return "❌ Error: API key doesn't start with sk-proj-. Please check you're using the right key."#If API key doesn't start with espected prefix
    elif api_key.strip() != api_key:
        #If API key has leading, trailing spaces/tabs
        return "❌ Error: An API key was found, but it looks like it might have space or tab characters at the start or end "
    
    print("API key found and looks good!") #Testing message
    
    if constrained_results==True:
        system_prompt = """
        You are a fact-checking assistant that analyzes the user prompt text and the site to fact-check
        and return yes or no as a answer with a short explanation and percentage-score of credibility.
        List the sources and links of fact sources as is. Credibility score should reflect percentage-accuracy of user-inputs.
         Only use the following two sites: site:factcheck.org AND/OR site:who.int
        """ #Simple prompt to tell model how to behave when user does want constrained results
    else:
         system_prompt = """
         You are a fact-checking assistant that analyzes the user prompt text and the site to fact-check
         and return yes or no as a answer with a short explanation and percentage-score of credibility.
         Credibility score should reflect percentage-accuracy of user-inputs.
         List the sources and links of fact sources as is. No limitation on credible sites to use.
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
        final_response = response.choices[0].message.content # Store the text response
        return final_response
    except Exception as e:
        return f"❌ Error: {str(e)}" # If any error occurs in the try block, return error message as string

def check_claim(claim, include_sites):
    #If claim is empty or whitespace, return warning
    if not claim.strip():
        return "⚠️ Please enter a claim to fact-check."
    
    # Add site restrictions if checkbox is selected
    if include_sites: #If user checked the checkbox
        constrained_results= True #constrained_results 'True' means: sysytem prompt limited to check sites: site:factcheck.org OR site:who.int
    else:
        constrained_results=False #constrained_results 'False' means: system prompt is NOT limtied specific sites 
    
    return fact_checker(claim, constrained_results) #Call fact_checker with the query and constrained_result boolean variable




#Speech-to-text translation 
#Code from Gradio
def transcribe_audio(audio):
    if audio is None:
        return ""
    sr, y = audio
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y})["text"]


#Fact Checking the transcribed speech:
# Fact-check the transcribed speech
def fact_check_transcribed(text, include_sites=True):
    if not text.strip():
        return "⚠️ No speech detected to fact-check."
    return check_claim(text, include_sites)


# Create Gradio Interface
with gr.Blocks(title="Fact Checker AI", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🔍 Fact Checker AI
        ### Verify claims and statements with AI-powered fact-checking
        Choose either text input or audio recording to check claims.
        """
    )

    # Row for the two buttons
    with gr.Row():
        text_tab_btn = gr.Button("💬 Text Input Only")
        audio_tab_btn = gr.Button("🎤 Audio Recording Only")

    # Text input section (hidden initially)
    with gr.Row(visible=False) as text_section:
        with gr.Column(scale=2):
            claim_input = gr.Textbox(label="Enter Claim to Fact-Check", placeholder="Example: Vaccines cause autism", lines=3)
            site_checkbox = gr.Checkbox(label="Search only trusted sources (factcheck.org, who.int)", value=True)
            with gr.Row():
                submit_btn = gr.Button("🔍 Check Fact", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", scale=1)
        output = gr.Markdown(label="Fact-Check Result", value="Results will appear here...")
        # Examples
        gr.Examples(
            examples=[
                ["Vaccines cause autism", True],
                ["The Earth is flat", True],
                ["Coffee is bad for your health", True],
                ["5G networks spread COVID-19", True],
            ],
            inputs=[claim_input, site_checkbox],
            label="Click any example to try it"
        )

    # Audio input section (hidden initially)
    with gr.Row(visible=False) as audio_section:
        audio_input = gr.Audio(label="Speak Here", type="numpy")
        transcribed_text = gr.Textbox(label="Transcribed Text", placeholder="Your speech will appear here...")
        check_speech_btn = gr.Button("🔍 Fact-Check Speech")
        fact_output_transcribed = gr.Markdown(label="Fact-Check Result")

    # Button events to toggle sections
    text_tab_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[text_section, audio_section])
    audio_tab_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[text_section, audio_section])

    # Text events
    submit_btn.click(fn=check_claim, inputs=[claim_input, site_checkbox], outputs=output)
    clear_btn.click(fn=lambda: ("", "Results will appear here..."), outputs=[claim_input, output])
    claim_input.submit(fn=check_claim, inputs=[claim_input, site_checkbox], outputs=output)

    # Audio events
    audio_input.change(fn=transcribe_audio, inputs=[audio_input], outputs=[transcribed_text])
    check_speech_btn.click(fn=fact_check_transcribed, inputs=[transcribed_text, site_checkbox], outputs=[fact_output_transcribed])


def main():
    # Launch Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share= True  
    )

if __name__ == "__main__":
    main()