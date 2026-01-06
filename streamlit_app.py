#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit application for MetaGPT chat interface.
Provides a user-friendly chat interface for interacting with MetaGPT.
"""
import asyncio
from typing import Optional

import streamlit as st

from metagpt.llm import LLM
from metagpt.logs import logger


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm" not in st.session_state:
        st.session_state.llm = None


async def get_llm_response(prompt: str, llm: LLM) -> str:
    """
    Get response from LLM asynchronously.
    
    Args:
        prompt: User input prompt
        llm: LLM instance
        
    Returns:
        Response string from LLM
    """
    try:
        response = await llm.aask(prompt, system_msgs=["You are a helpful AI assistant."])
        return response
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


def main():
    """Main Streamlit application."""
    st.title("MetaGPT Chat Interface")
    st.caption("A chat interface powered by MetaGPT")
    
    # Initialize session state
    init_session_state()
    
    # Initialize LLM if not already done
    if st.session_state.llm is None:
        with st.spinner("Initializing MetaGPT..."):
            try:
                st.session_state.llm = LLM()
                st.success("MetaGPT initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize MetaGPT: {str(e)}")
                st.stop()
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                try:
                    # Run async function in event loop
                    response = asyncio.run(
                        get_llm_response(prompt, st.session_state.llm)
                    )
                    message_placeholder.markdown(response)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    response = error_msg
                    logger.error(f"Error in chat: {e}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with additional options
    with st.sidebar:
        st.header("Options")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This is a chat interface for MetaGPT. "
            "Ask questions and get AI-powered responses."
        )


if __name__ == "__main__":
    main()
