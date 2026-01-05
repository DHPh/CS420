"""
Launch Streamlit app for B-Free AI Image Detection
"""
import subprocess
import sys

if __name__ == "__main__":
    print("Starting B-Free Streamlit App...")
    print("=" * 60)
    print("The app will open in your browser automatically")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])
