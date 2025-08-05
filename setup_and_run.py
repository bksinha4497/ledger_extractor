import subprocess
import sys
import os
import platform

def install_system_dependencies():
    """Install system-level dependencies based on the operating system"""
    system = platform.system().lower()
    
    print("Installing system dependencies...")
    
    if system == "darwin":  # macOS
        print("Detected macOS. Installing dependencies via Homebrew...")
        try:
            # Install Tesseract OCR
            subprocess.run(["brew", "install", "tesseract"], check=True)
            # Install poppler for pdf2image
            subprocess.run(["brew", "install", "poppler"], check=True)
            print("System dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Error installing system dependencies. Please install manually:")
            print("brew install tesseract poppler")
            return False
        except FileNotFoundError:
            print("Homebrew not found. Please install Homebrew first:")
            print('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
            print("Then run: brew install tesseract poppler")
            return False
    
    elif system == "linux":
        print("Detected Linux. Installing dependencies via apt...")
        try:
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "tesseract-ocr", "poppler-utils"], check=True)
            print("System dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Error installing system dependencies. Please install manually:")
            print("sudo apt update && sudo apt install -y tesseract-ocr poppler-utils")
            return False
    
    elif system == "windows":
        print("Detected Windows. Please install dependencies manually:")
        print("1. Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install poppler from: https://blog.alivate.com.au/poppler-windows/")
        print("3. Add both to your system PATH")
        return False
    
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Python dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Error installing Python dependencies. Please run manually:")
        print("pip install -r requirements.txt")
        return False

def run_extractor():
    """Run the PDF ledger extractor"""
    print("\nRunning PDF Ledger Extractor...")
    try:
        subprocess.run([sys.executable, "pdf_ledger_extractor.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running extractor: {e}")
        return False
    return True

def main():
    print("PDF Ledger Extractor Setup and Run Script")
    print("=" * 50)
    
    # Check if required files exist
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found!")
        return
    
    if not os.path.exists("pdf_ledger_extractor.py"):
        print("Error: pdf_ledger_extractor.py not found!")
        return
    
    if not os.path.exists("data_pdf"):
        print("Error: data_pdf directory not found!")
        return
    
    # Install system dependencies
    if not install_system_dependencies():
        print("\nPlease install system dependencies manually and run the script again.")
        return
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\nPlease install Python dependencies manually and run the script again.")
        return
    
    # Run the extractor
    print("\nSetup complete! Running the extractor...")
    run_extractor()

if __name__ == "__main__":
    main()
