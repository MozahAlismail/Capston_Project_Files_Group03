# app.py
import subprocess
import time

def main():
    # Start the FastAPI server in the background
    api_process = subprocess.Popen([
        "uvicorn", "Rag_Application_Files.api:app", "--port", "8000"
    ])

    # Wait a few seconds to ensure the API is up
    time.sleep(3)

    # Start the Chainlit interface
    chainlit_process = subprocess.Popen([
        "chainlit", "run", "Rag_Application_Files/main.py", "--port", "8001"
    ])

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
