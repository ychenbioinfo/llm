from httpx import AsyncClient
from datetime import datetime
import streamlit as st
import asyncio
from typing import Any
import json
import os

from pydantic_ai.messages import ModelResponse, ModelRequest, UserPromptPart, TextPart, ModelMessage

from agent import agent, Deps


async def prompt_ai(messages: list[ModelMessage]):
    async with AsyncClient() as client:
        deps = Deps(client=client)
        async with agent.run_stream(
            messages[-1].parts[0].content, deps=deps, message_history=messages[:-1]
        ) as result:
            async for message in result.stream_text(delta=True):
                yield message


async def main():
    st.title("Pydantic AI Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if isinstance(message, ModelRequest):
            with st.chat_message("human"):
                st.markdown(message.parts[0].content)
        elif isinstance(message, ModelResponse):
            with st.chat_message("assistant"):
                st.markdown(message.parts[0].content)

    # React to user input
    if prompt := st.chat_input("What question would you like to ask?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(ModelRequest(parts=[UserPromptPart(content=prompt)]))

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            # Run the async generator to fetch responses
            async for chunk in prompt_ai(st.session_state.messages):
                response_content += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)
        
        if not response_content:
            response_content = "Sorry, I don't know the answer to that question."
            message_placeholder.markdown(response_content)
        st.session_state.messages.append(ModelResponse(parts=[TextPart(content=response_content)]))


if __name__ == "__main__":
    asyncio.run(main())