# app.py
import subprocess
import time
import os
import signal
import sys

def main():
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting services on port {port}")
    
    # Start the FastAPI server in the background
    api_process = subprocess.Popen([
        "uvicorn", "Rag_Application_Files.api:app", 
        "--host", "0.0.0.0", 
        "--port", str(port)
    ])
    
    print(f"✅ FastAPI server started on port {port}")

    # Wait a few seconds to ensure the API is up
    time.sleep(3)

    # Start the Chainlit interface on a different port
    chainlit_port = port + 1
    chainlit_process = subprocess.Popen([
        "chainlit", "run", "Rag_Application_Files/main.py", 
        "--host", "0.0.0.0",
        "--port", str(chainlit_port)
    ])
    
    print(f"✅ Chainlit interface started on port {chainlit_port}")

    def signal_handler(sig, frame):
        print("🔴 Shutting down gracefully...")
        api_process.terminate()
        chainlit_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Wait for both to finish
        api_process.wait()
        chainlit_process.wait()
    except KeyboardInterrupt:
        print("🔴 Shutting down...")
        api_process.terminate()
        chainlit_process.terminate()

if __name__ == "__main__":
    main()
