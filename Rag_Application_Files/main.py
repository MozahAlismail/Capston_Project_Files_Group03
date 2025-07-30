import chainlit as cl
import httpx
import os
import asyncio
import subprocess
import time
import sys
import threading
import uvicorn
from multiprocessing import Process
import signal
import atexit

# Set your Hugging Face token directly here (REPLACE WITH YOUR ACTUAL TOKEN)
from huggingface_hub import login
# os.environ["HUGGINGFACE_API_TOKEN"] = "Token_Here"  # Replace with your actual token
login(token=os.environ["HUGGINGFACE_API_TOKEN"])

# Configuration - Dynamic based on environment
def get_environment_config():
    
    # Local development
    host = "127.0.0.1"
    api_port = 8000
    chainlit_port = 8001
    api_url = os.getenv("FASTAPI_URL", f"http://localhost:{api_port}/chat")
    
    return host, api_port, chainlit_port, api_url, "Development"

# Get dynamic configuration
FASTAPI_HOST, FASTAPI_PORT, CHAINLIT_PORT, FASTAPI_URL, ENVIRONMENT_TYPE = get_environment_config()

# Global variables for server management
api_server_process = None
api_server_thread = None

def check_api_server():
    """Check if the API server is running"""
    try:
        import requests
        response = requests.get(f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the FastAPI server in a separate process"""
    global api_server_process
    
    if check_api_server():
        print("✅ API server already running")
        return True
    
    print("🚀 Starting FastAPI server...")
    
    try:
        # Add current directory to Python path for imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import and start the FastAPI app
        from api import app
        
        def run_server():
            uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT, log_level="info")
        
        # Start server in a separate thread
        global api_server_thread
        api_server_thread = threading.Thread(target=run_server, daemon=True)
        api_server_thread.start()
        
        # Wait for server to start
        for i in range(30):  # Wait up to 30 seconds
            if check_api_server():
                print("✅ FastAPI server started successfully")
                return True
            time.sleep(1)
            print(f"⏳ Waiting for API server to start... ({i+1}/30)")
        
        print("❌ Failed to start API server within 30 seconds")
        return False
        
    except ImportError as e:
        print(f"❌ Could not import API module: {e}")
        print("⚠️  Make sure 'api.py' exists in the same directory as main.py")
        return False
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return False

def stop_api_server():
    """Stop the API server"""
    global api_server_process, api_server_thread
    
    if api_server_process:
        api_server_process.terminate()
        api_server_process = None
        print("🛑 API server process terminated")
    
    if api_server_thread:
        # Thread will be cleaned up automatically since it's a daemon thread
        api_server_thread = None
        print("🛑 API server thread stopped")

# Register cleanup function
atexit.register(stop_api_server)

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Local development - start API server if not running
    if not check_api_server():
        print("🔄 API server not detected, starting...")
        if start_api_server():
            await cl.Message(
                content="🎉 **Welcome to the AI Policy Assistant!**\n\n"
                        "✅ Backend API server is running\n"
                        "✅ RAG system is ready\n\n"
                        "Ask me anything about AI governance, policies, or data management!"
            ).send()
        else:
            await cl.Message(
                content="⚠️ **Warning: Backend API server failed to start**\n\n"
                        "The chat interface is ready, but the AI backend may not be available.\n\n"
                        "**Possible solutions:**\n"
                        "1. Make sure `api.py` exists in the same directory\n"
                        "2. Start the API server manually: `python api.py`\n"
                        "3. Check the console for detailed error messages\n\n"
                        "*You can still use this interface, but responses may show connection errors.*"
            ).send()
    else:
        await cl.Message(
            content="🎉 **Welcome to the AI Policy Assistant!**\n\n"
                    "✅ Connected to running backend API\n"
                    "✅ RAG system is ready\n\n"
                    "Ask me anything about AI governance, policies, or data management!"
        ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming messages"""
    # Show loading message
    loading_msg = cl.Message(content="🤔 Thinking...")
    await loading_msg.send()
    
    # Add debug logging
    print(f"🔍 Received message: {message.content}")
    print(f"🌐 API URL: {FASTAPI_URL}")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout for model loading
            print(f"📡 Sending request to: {FASTAPI_URL}")
            
            # Try the new API format first
            response = await client.post(FASTAPI_URL, json={"question": message.content})
            
            print(f"📨 Response status: {response.status_code}")
            print(f"📋 Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Response data keys: {list(data.keys())}")
                
                if "answer" in data:
                    answer = data["answer"]
                    print(f"📝 Answer length: {len(answer)} characters")
                elif "error" in data:
                    answer = f"❌ API Error: {data['error']}"
                    print(f"⚠️  API returned error: {data['error']}")
                else:
                    answer = f"❌ Unexpected response format: {data}"
                    print(f"🔍 Full response data: {data}")
            else:
                answer = f"❌ API Error: {response.status_code} - {response.text}"
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"📄 Response text: {response.text[:200]}...")
                
    except httpx.ConnectError as e:
        print(f"🔌 Connection error: {e}")
        answer = (f"❌ **Connection Error**\n\n"
                 f"Cannot connect to the API server at {FASTAPI_URL}\n\n"
                 f"**Troubleshooting:**\n"
                 f"1. API server may be starting up (please wait)\n"
                 f"2. Check if the server is accessible at {FASTAPI_URL}\n"
                 f"3. Verify no firewall is blocking the connection\n\n"
                 f"*The API server should start automatically. Please try again in a few seconds.*")
    except httpx.TimeoutException as e:
        print(f"⏱️ Timeout error: {e}")
        answer = ("⏱️ **Timeout Error**\n\n"
                 "The API server took too long to respond (>120s).\n"
                 "This usually happens when the AI model is loading for the first time.\n\n"
                 "**Please try again** - subsequent requests should be faster.")
    except Exception as e:
        print(f"🔥 Unexpected error: {e}")
        print(f"📊 Error type: {type(e).__name__}")
        answer = f"🔥 **Unexpected Error**\n\n{str(e)}\n\nPlease try again or contact support if the issue persists."
    
    print(f"📤 Sending final answer: {answer[:100]}...")
    
    # Update the loading message with the response
    try:
        await loading_msg.update(content=answer)
        print("✅ Message updated successfully")
    except Exception as e:
        print(f"❌ Failed to update message: {e}")
        # Fallback: send a new message
        await cl.Message(content=answer).send()

def main():
    """Main function to start the complete system"""
    print("=" * 60)
    print("🚀 llm chainLIT API - MAIN LAUNCHER")
    print("=" * 60)
    
    print(f"🔧 Configuration:")
    print(f"   Environment Type: {ENVIRONMENT_TYPE}")
    print(f"   FastAPI URL: {FASTAPI_URL}")
    print(f"   FastAPI Host: {FASTAPI_HOST}:{FASTAPI_PORT}")
    print(f"   Chainlit Port: {CHAINLIT_PORT}")
    print()
    
    print("💻 Local development mode")
    print("   - Will manage API server automatically")
    print("   - Using localhost (127.0.0.1) for local access")
    print("   - API and Chainlit on separate ports")
    
    # Pre-start API server for faster first response
    if start_api_server():
        print("✅ API server pre-started successfully")
    else:
        print("⚠️  API server will be started when needed")
    
    print()
    print(f"🌐 Starting Chainlit interface on {FASTAPI_HOST}:{CHAINLIT_PORT}...")
    print("=" * 60)

if __name__ == "__main__":
    main()