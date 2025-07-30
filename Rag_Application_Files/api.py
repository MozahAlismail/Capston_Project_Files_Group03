from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn
import nest_asyncio
import os
import traceback
from rag import rag_chat, initialize_rag_system


# Set your Hugging Face token directly here (REPLACE WITH YOUR ACTUAL TOKEN)
from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_API_TOKEN"])

# Define request model for better validation
class ChatRequest(BaseModel):
    question: str

# Initialize the RAG function with proper error handling
rag_initialized = False

def initialize_rag():
    """Initialize RAG system on first request"""
    global rag_initialized
    
    if rag_initialized:
        return True
    
    try:
        
        print("Loading original RAG implementation...")
        # Functions are already imported at the top
        print("✅ Original RAG loaded successfully")
        
        # Initialize the system
        initialize_rag_system()
            
        rag_initialized = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading RAG implementation: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

try:
    # Try to import and initialize on startup (for faster first response)
    initialize_rag()
except Exception as e:
    print(f"⚠️  Could not initialize RAG on startup: {e}")
    print("RAG will be initialized on first request.")

app = FastAPI(title="llm chainlit API", description="AI Policy Assistant API")

@app.get("/")
async def health_check():
    return {
        "status": "healthy", 
        "service": "llm chainlit API", 
        "version": "1.0.0",
        "rag_loaded": True,  # Functions are imported at module level
        "rag_initialized": rag_initialized,
        "environment": "development"
    }

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        # Initialize RAG if not already done
        if not rag_initialized:
            print("🔄 Initializing RAG system on first request...")
            if not initialize_rag():
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to initialize RAG system. Check server logs."
                )
        
        question = chat_request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"Processing question: {question}")
        
        # Call the RAG function
        answer = rag_chat(question)
        
        if not answer:
            answer = "I apologize, but I couldn't generate a response. Please try again."
        
        print(f"Generated answer: {answer[:100]}...")
        
        return {"answer": answer}
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)