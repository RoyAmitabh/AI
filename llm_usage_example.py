"""
LLM Direct Usage Examples
When to use LLMs without additional frameworks or techniques.
"""

from src.util.commons import get_llm_instance

def conversation_examples():
    """Basic conversation examples."""
    llm = get_llm_instance()
    
    conversation = [
        "Hello, how are you?",
        "What's the weather like today in Bangalore?",        
        "Tell me a joke"
    ]
          
    for message in conversation:
        print(f"\n User: {message}")
        response = llm.invoke(message)
        print(f" Assistant: {response.content}")


if __name__ == "__main__":   
    conversation_examples()
    
