import gradio as gr
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def chat(message: str, chat_history: list[str]):
    messages = []

    for msg in chat_history:
        messages.append({
            "role": "user",
            "content": msg[0]
        })
        messages.append({
            "role": "assistant",
            "content": msg[1]
        })        

    messages.append({
        "role": "user",
        "content": message
    })
        
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    

    chat_history.append([message, "Thinking..."])

    yield "", chat_history

    chat_history[-1][1] = ""
    for chunk in chat_completion:
        delta = chunk.choices[0].delta.content or ""
        chat_history[-1][1] += delta
        yield "", chat_history

    return "", chat_history        


with gr.Blocks() as demo:
    gr.Markdown("# Gradio Chatbot with OpenAI")
    message = gr.Textbox(label="Enter your message")
    chatbot = gr.Chatbot(label="Super Chatbot AI")

    message.submit(chat, inputs=[message, chatbot], outputs=[message, chatbot])

demo.launch()