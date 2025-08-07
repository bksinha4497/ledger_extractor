#!/usr/bin/env python3
"""
Setup script for AI OCR Bank Statement Extractor
Installs dependencies and tests the setup
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_ai_ocr.txt"])
        print("✅ Python dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False
    return True

def check_system_dependencies():
    """Check if system dependencies are installed"""
    print("🔍 Checking system dependencies...")
    
    # Check Tesseract
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        print("✅ Tesseract OCR is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Tesseract OCR not found. Install with:")
        print("   macOS: brew install tesseract")
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        return False
    
    # Check Poppler (for pdf2image)
    try:
        subprocess.run(["pdftoppm", "-h"], capture_output=True, check=True)
        print("✅ Poppler is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poppler not found. Install with:")
        print("   macOS: brew install poppler")
        print("   Ubuntu/Debian: sudo apt-get install poppler-utils")
        return False
    
    return True

def test_ocr_engines():
    """Test available OCR engines"""
    print("🧪 Testing OCR engines...")
    
    # Test basic imports
    try:
        import pytesseract
        print("✅ Tesseract OCR available")
    except ImportError:
        print("❌ Tesseract OCR not available")
    
    try:
        import easyocr
        print("✅ EasyOCR available")
    except ImportError:
        print("⚠️  EasyOCR not available (optional)")
    
    try:
        from google.cloud import vision
        print("✅ Google Vision API available")
    except ImportError:
        print("⚠️  Google Vision API not available (optional)")

def create_sample_usage():
    """Create a sample usage script"""
    sample_code = '''#!/usr/bin/env python3
"""
Sample usage of AI OCR Bank Statement Extractor
"""

from ai_ocr_extractor import AIBankStatementExtractor
import os

def main():
    # Initialize the extractor
    pdf_directory = "data_pdf"  # Directory containing your PDF bank statements
    output_file = "extracted_transactions.csv"
    
    extractor = AIBankStatementExtractor(pdf_directory, output_file)
    
    # Process all PDFs with automatic OCR method selection
    print("🔄 Processing bank statement PDFs...")
    df = extractor.process_all_pdfs(ocr_method="auto")
    
    if not df.empty:
        print(f"✅ Successfully extracted {len(df)} transactions")
        print(f"📄 Data saved to: {output_file}")
        
        # Generate summary report
        report = extractor.generate_report(df)
        report_file = output_file.replace('.csv', '_report.md')
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📊 Report saved to: {report_file}")
        
        # Display first few transactions
        print("\\n📋 Sample transactions:")
        print(df.head().to_string(index=False))
    else:
        print("❌ No transactions were extracted")

if __name__ == "__main__":
    main()
'''
    
    with open("sample_usage.py", "w") as f:
        f.write(sample_code)
    print("📝 Created sample_usage.py")

def main():
    """Main setup function"""
    print("🚀 Setting up AI OCR Bank Statement Extractor")
    print("=" * 50)
    
    # Check system dependencies first
    if not check_system_dependencies():
        print("\\n❌ Please install missing system dependencies before continuing")
        return
    
    # Install Python requirements
    if not install_requirements():
        print("\\n❌ Setup failed due to dependency installation issues")
        return
    
    # Test OCR engines
    test_ocr_engines()
    
    # Create sample usage
    create_sample_usage()
    
    print("\\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\\n📖 Usage:")
    print("1. Place your PDF bank statements in the 'data_pdf' directory")
    print("2. Run: python ai_ocr_extractor.py --pdf-dir data_pdf --output results.csv")
    print("3. Or run the sample: python sample_usage.py")
    print("\\n🔧 OCR Methods available:")
    print("- auto: Automatically selects best OCR method")
    print("- tesseract: Traditional OCR (fastest)")
    print("- easyocr: Deep learning OCR (more accurate)")
    print("- google_vision: Google Cloud Vision API (most accurate, requires setup)")

if __name__ == "__main__":
    main()
