"""
RAG Usage Examples
When to use Retrieval-Augmented Generation.
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings  # Use Azure-specific embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.util.commons import get_llm_instance

def create_knowledge_base():
    """Create different types of knowledge bases for RAG."""
    
    # Company documentation
    company_docs = [
        "Our company policy states that employees get 25 days of vacation per year.",
        "The dress code is business casual on weekdays and casual on Fridays.",
        "Health insurance covers 80% of medical expenses for employees and families.",
        "Remote work is allowed up to 3 days per week with manager approval.",
        "Performance reviews are conducted quarterly with annual salary adjustments."
    ]
    
    # Product documentation
    product_docs = [
        "Our software supports Windows 10, 11, and macOS 12.0+.",
        "The API rate limit is 1000 requests per hour per user.",
        "Data is encrypted using AES-256 encryption at rest and in transit.",
        "Backup is performed automatically every 24 hours.",
        "Customer support is available 24/7 via email and phone."
    ]
    
    # Legal documents
    legal_docs = [
        "The contract termination requires 30 days written notice.",
        "Intellectual property created during employment belongs to the company.",
        "Non-disclosure agreements are valid for 5 years after employment.",
        "Severance pay is calculated as 2 weeks per year of service.",
        "Disputes are resolved through binding arbitration in the state of California."
    ]
    
    return {
        "company": company_docs,
        "product": product_docs,
        "legal": legal_docs
    }

def demonstrate_rag_use_cases():
    """Demonstrate different RAG use cases."""
    
    print("🔍 RAG Usage Examples")
    print("=" * 50)
    
    knowledge_bases = create_knowledge_base()
    
    for kb_type, documents in knowledge_bases.items():
        print(f"\n📚 {kb_type.upper()} KNOWLEDGE BASE")
        print("-" * 40)
        
        # Create RAG system for this knowledge base
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        docs = [Document(page_content=text) for text in documents]
        splits = text_splitter.split_documents(docs)
        
        # Use text-embedding model
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",  # Use text embedding model
            openai_api_version="2024-02-15-preview",
            openai_api_key="",
            azure_endpoint=""
        )
        
        vectorstore = Chroma.from_documents(splits, embeddings)
        
        llm = get_llm_instance()
        
        # Create RAG chain using newer patterns
        template = """Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        qa_chain = (
            {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Example questions for each knowledge base
        questions = {
            "company": [
                "How many vacation days do employees get?",
                "What is the dress code policy?",
                "Is remote work allowed?"
            ],
            "product": [
                "What operating systems are supported?",
                "What is the API rate limit?",
                "How often are backups performed?"
            ],
            "legal": [
                "How much notice is required for contract termination?",
                "Who owns intellectual property created during employment?",
                "How long are NDAs valid?"
            ]
        }
        
        for question in questions[kb_type]:
            print(f"\n❓ Question: {question}")
            answer = qa_chain.invoke(question)
            print(f"🤖 Answer: {answer}")

def real_world_rag_scenarios():
    """Show real-world scenarios where RAG is essential."""
    
    print("\n" + "=" * 60)
    print("🌍 Real-World RAG Scenarios")
    print("=" * 60)
    
    scenarios = {
        "Customer Support": {
            "Problem": "Customers ask about product features, policies, and troubleshooting",
            "Solution": "RAG with product documentation and FAQ database",
            "Benefit": "Accurate, up-to-date answers without training new models"
        },
        "Legal Research": {
            "Problem": "Lawyers need to search through case law and regulations",
            "Solution": "RAG with legal document database",
            "Benefit": "Finds relevant precedents and citations quickly"
        },
        "Medical Diagnosis": {
            "Problem": "Doctors need to reference medical literature and guidelines",
            "Solution": "RAG with medical journals and clinical guidelines",
            "Benefit": "Provides evidence-based recommendations"
        },
        "Academic Research": {
            "Problem": "Researchers need to find relevant papers and citations",
            "Solution": "RAG with academic paper database",
            "Benefit": "Discovers related research and current state of knowledge"
        },
        "Enterprise Knowledge": {
            "Problem": "Employees need to find company policies and procedures",
            "Solution": "RAG with internal documentation",
            "Benefit": "Ensures compliance and consistent information"
        }
    }
    
    for scenario, details in scenarios.items():
        print(f"\n🎯 {scenario}:")
        print(f"   Problem: {details['Problem']}")
        print(f"   Solution: {details['Solution']}")
        print(f"   Benefit: {details['Benefit']}")

def when_to_use_rag():
    """Explain when to use RAG."""
    
    print("\n" + "=" * 60)
    print("✅ When to Use RAG")
    print("=" * 60)
    
    use_cases = {
        "Specific Knowledge": "Questions requiring domain-specific information",
        "Real-time Data": "Information that changes frequently",
        "Document Q&A": "Questions about specific documents or policies",
        "Factual Accuracy": "When you need verifiable, sourced information",
        "Large Knowledge Bases": "When information is too large for LLM context",
        "Compliance": "When answers must be based on official documentation",
        "Multi-source Information": "When information comes from multiple sources"
    }
    
    for use_case, description in use_cases.items():
        print(f"📋 {use_case}: {description}")

if __name__ == "__main__":
    demonstrate_rag_use_cases()
    real_world_rag_scenarios()
    when_to_use_rag() 
