"""
Common utilities for the AI
"""

from langchain_openai import AzureChatOpenAI

def get_llm_instance():
    """Get an instance of the LLM configured for Azure OpenAI."""
    llm = AzureChatOpenAI(
        azure_deployment="<your deployment name in Azure>",
        temperature=0.5,
        azure_endpoint="<your azure endpoint for openai>",  
        openai_api_version="2024-02-15-preview",
        openai_api_key="...",
        seed=50  
    )
    return llm
