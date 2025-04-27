import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Agent
greeting_agent = Agent(
    model=model,
    instructions="You are a helpful assistant that greets the user.",
    name="greeting_agent"
)

# Chainlit provide history of messages
@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    


@cl.on_message  # decorator function
async def main(message: cl.Message):
    
    # Show something on the screen
    msg = cl.Message(
        content="Thinking...",
    )
    await msg.send()

    # Step 1: 
    print("\nStep 1:Get History and add User Message\n")
    history = cl.user_session.get("history") # [...]
    print("History: ", history)
    print("\nStep 2: Add User Messaged to History\n")
    history.append({"role": "user", "content": message.content}) # [{}]    
    print("Updated History: ", history)

    # Agent Call
    agent_response = await Runner.run(greeting_agent, history)
    msg.content = agent_response.final_output
    await msg.update()
    
    # Step 2:
    # Get History and add Agent Message
    print("\nStep 3: Get History and add Agent Message\n")
    history.append({"role": "assistant", "content": agent_response.final_output})
    # Step 3:
    # Update History
    print("\nStep 4: Update History\n")
    cl.user_session.set("history", history)
