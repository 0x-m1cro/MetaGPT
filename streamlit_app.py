#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit application for MetaGPT chat interface.
Provides a user-friendly chat interface for interacting with MetaGPT.
"""
import asyncio

import streamlit as st

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.llm import LLM
from metagpt.logs import logger

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

# Common OpenAI models
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1-preview",
    "o1-mini",
]

# Model configurations with their default settings
MODEL_CONFIGS = {
    "OpenAI": {
        "api_type": LLMType.OPENAI,
        "base_url": "https://api.openai.com/v1",
        "models": OPENAI_MODELS,
        "default_model": "gpt-4o-mini",
    },
    "Azure OpenAI": {
        "api_type": LLMType.AZURE,
        "base_url": "https://YOUR_RESOURCE_NAME.openai.azure.com",
        "models": ["gpt-4", "gpt-35-turbo"],
        "default_model": "gpt-4",
        "requires_api_version": True,
    },
    "Anthropic": {
        "api_type": LLMType.ANTHROPIC,
        "base_url": "https://api.anthropic.com",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "default_model": "claude-3-5-sonnet-20241022",
    },
    "Ollama": {
        "api_type": LLMType.OLLAMA,
        "base_url": "http://localhost:11434",
        "models": ["llama3.1", "llama3.2", "mistral", "phi3"],
        "default_model": "llama3.1",
    },
}


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = "OpenAI"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o-mini"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "base_url" not in st.session_state:
        st.session_state.base_url = "https://api.openai.com/v1"
    if "llm_initialized" not in st.session_state:
        st.session_state.llm_initialized = False


def create_llm_config(provider: str, model: str, api_key: str, base_url: str, api_version: str = None) -> LLMConfig:
    """
    Create LLM configuration based on selected provider and model.
    
    Args:
        provider: Selected provider name
        model: Selected model name
        api_key: API key for the provider
        base_url: Base URL for the API
        api_version: API version (for Azure)
        
    Returns:
        LLMConfig instance
    """
    config_template = MODEL_CONFIGS[provider]
    
    config_dict = {
        "api_type": config_template["api_type"],
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }
    
    if config_template.get("requires_api_version") and api_version:
        config_dict["api_version"] = api_version
    
    return LLMConfig(**config_dict)


async def get_llm_response(prompt: str, llm: LLM, system_prompt: str) -> str:
    """
    Get response from LLM asynchronously.
    
    Args:
        prompt: User input prompt
        llm: LLM instance
        system_prompt: System prompt for the LLM
        
    Returns:
        Response string from LLM
    """
    try:
        response = await llm.aask(prompt, system_msgs=[system_prompt])
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
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🤖 Model Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "AI Provider",
            options=list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS.keys()).index(st.session_state.selected_provider),
            help="Select your AI provider",
        )
        
        # Update models list based on provider
        provider_config = MODEL_CONFIGS[provider]
        
        # Model selection
        model = st.selectbox(
            "Model",
            options=provider_config["models"],
            index=provider_config["models"].index(st.session_state.selected_model)
            if st.session_state.selected_model in provider_config["models"]
            else 0,
            help="Select the AI model to use",
        )
        
        # API Key input
        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your API key for the selected provider",
        )
        
        # Base URL input
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.base_url if st.session_state.base_url else provider_config["base_url"],
            help="API endpoint URL",
        )
        
        # API Version (for Azure)
        api_version = None
        if provider_config.get("requires_api_version"):
            api_version = st.text_input(
                "API Version",
                value="2024-02-15-preview",
                help="Azure OpenAI API version",
            )
        
        # Initialize/Reinitialize LLM button
        if st.button("🔄 Initialize LLM", type="primary"):
            if not api_key:
                st.error("Please provide an API key!")
            else:
                with st.spinner("Initializing LLM..."):
                    try:
                        llm_config = create_llm_config(provider, model, api_key, base_url, api_version)
                        st.session_state.llm = LLM(llm_config=llm_config)
                        st.session_state.selected_provider = provider
                        st.session_state.selected_model = model
                        st.session_state.api_key = api_key
                        st.session_state.base_url = base_url
                        st.session_state.llm_initialized = True
                        st.success(f"✅ Initialized {provider} - {model}")
                    except Exception as e:
                        st.error(f"Failed to initialize: {str(e)}")
                        st.session_state.llm_initialized = False
        
        st.markdown("---")
        st.header("💬 Chat Options")
        
        # System prompt configuration
        new_system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            help="Customize how the AI assistant behaves",
            height=100,
        )
        if new_system_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = new_system_prompt
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This is a chat interface for MetaGPT. "
            "Configure your preferred AI model and start chatting!"
        )
        
        if st.session_state.llm_initialized:
            st.success(f"Current: {st.session_state.selected_provider} - {st.session_state.selected_model}")
    
    # Main chat interface
    if not st.session_state.llm_initialized:
        st.info("👈 Please configure and initialize the LLM in the sidebar to start chatting.")
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
                    # Run async function - asyncio.run() is the standard approach in Streamlit
                    response = asyncio.run(
                        get_llm_response(prompt, st.session_state.llm, st.session_state.system_prompt)
                    )
                    message_placeholder.markdown(response)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    response = error_msg
                    logger.error(f"Error in chat: {e}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
