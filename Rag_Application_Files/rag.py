from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from huggingface_hub import hf_hub_download
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import zipfile
from transformers import pipeline
import torch
import os
from together import Together

# Set your Hugging Face token directly here (REPLACE WITH YOUR ACTUAL TOKEN)
from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_API_TOKEN"])

# Global variables to store initialized components
embedding_model = None
vectorstore = None
retriever = None
llm = None
llm_chain = None
chat_history = []

def initialize_embeddings():
    """Initialize the embedding model"""
    global embedding_model
    print("🔄 Initializing embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    print("✅ Embeddings initialized successfully")
    return embedding_model

def initialize_vectorstore():
    """Initialize the vector database"""
    global vectorstore, retriever
    print("🔄 Loading ChromaDB...")

    # Download and load the Chroma database
    def load_vector_db():
        chroma_db_path = hf_hub_download(
            repo_id="Ola1mohammed/GovernAI_dataset",
            filename="chroma_db.zip",
            repo_type="dataset",
        )
        
        extract_dir = "./chroma_db"
        
        # Only extract if folder is empty or does not exist
        if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(chroma_db_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        
        vector_db_path = os.path.join(extract_dir, "chroma_db")

        return Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)

    vector_db = load_vector_db()
    
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("✅ ChromaDB loaded successfully")
    return vector_db, retriever
        
def initialize_llm():
    """Initialize the Together AI hosted LLaMA 3.3 70B model"""
    global llm
    print("🔄 Loading Together AI model (via API)...")
    print("⚠️  Requires internet access and a Together API key.")

    try:
        # Set your Together API key
        client = Together(api_key=os.environ["TOGETHER_API_KEY"])

        # Model name hosted on Together
        model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        print(f"📦 Using Together AI model: {model_name}")

        # Define and assign the LLM function globally
        def _llm(messages):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()

        # Set the global llm to the defined function
        llm = _llm
        globals()["llm"] = llm

    except Exception as e:
        print(f"❌ Error initializing Together AI model: {e}")
        print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        raise

def create_llm_chain():
    """Create the LLM chain"""
    global llm_chain, prompt
    print("🔄 Creating LLM chain...")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI-powered policy assistant specialized in AI and Data Governance.
        Your role is to support users by interpreting and explaining governance frameworks, ethical standards, compliance guidelines, and policy definitions.

        Instructions:
        1. Base your answer only on the provided context.
        2. List the filenames of the documents you used (e.g., 'AI_Principles Document') under the "Sources" section.
        3. If the context does not contain the answer, respond with exactly: "I don't know."
        4. Do not make assumptions or add any information not explicitly stated in the context.

        Question: {question}

        Context: {context}

        Answer:
        """
    )
    
    prompt_1 = str(prompt.template)
    
    def build_messages(inputs):
        context = inputs["context"]
        question = inputs["question"]
        
        message = [
            {"role": "system", "content": prompt_1},
            *chat_history,  # ← this will hold prior conversation turns
            {"role": "user", "content": f"{question}\n\nContext:\n{context}"}
        ]
        
        return message

    def chat_with_memory(inputs):
        context = inputs["context"]
        question = inputs["question"]
        messages = build_messages(inputs)
        
        # Call the LLM
        response = llm(messages)

        # Save current turn to memory
        chat_history.append({"role": "user", "content": f"{question}\n\nContext:\n{context}"})
        chat_history.append({"role": "assistant", "content": response})

        return response
    
    llm_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(chat_with_memory)
        | StrOutputParser()
    ) 
    
    print("✅ llm chain created successfully")
    return llm_chain

def initialize_rag_system():
    """Initialize the entire RAG system"""
    print("🚀 Initializing RAG System...")
    print("=" * 50)
    
    try:
        # Initialize components in order
        initialize_embeddings()
        initialize_vectorstore()
        initialize_llm()
        create_llm_chain()
        
        print("=" * 50)
        print("✅ RAG System initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("=" * 50)
        return False

def rag_chat(user_message: str) -> str:
    """
    Main function to handle RAG chat requests
    
    Args:
        user_message (str): The user's question
        
    Returns:
        str: The generated response
    """
    try:
        if llm_chain is None:
            return "❌ Error: RAG system not initialized. Please restart the server."
        
        print(f"🔍 Processing question: {user_message}")
        print(f"📊 LLM chain initialized: {llm_chain is not None}")
        print(f"📊 Retriever initialized: {retriever is not None}")
        print(f"📊 LLM initialized: {llm is not None}")
        
        # Invoke the LLM chain
        print("🔄 Invoking LLM chain...")
        response = llm_chain.invoke(input=user_message)
        
        print(f"✅ Response generated successfully")
        print(f"📏 Response length: {len(response) if response else 0} characters")
        print(f"🔍 Response preview: {response[:100] if response else 'None'}...")
        
        if not response or response.strip() == "":
            return "I apologize, but I couldn't generate a response to your question. Please try rephrasing your question or try again."
        
        return response
        
    except Exception as e:
        print(f"❌ Error in rag_chat: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        print(f"📊 Stack trace: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

def test_rag_system():
    """Test the RAG system with a sample question"""
    print("\n🧪 Testing RAG system...")
    test_question = "What is AI governance?"
    response = rag_chat(test_question)
    print(f"Test Question: {test_question}")
    print(f"Response: {response[:200]}...")
    return response

def main():
    """Main function to initialize and test the RAG system"""
    print("🎯 Starting RAG System Setup...")
    
    # Initialize the RAG system
    if initialize_rag_system():
        # Test the system
        test_rag_system()
        print("\n🎉 RAG system is ready for use!")
    else:
        print("\n💥 Failed to initialize RAG system!")
        return False
    
    return True

if __name__ == "__main__":
    main()