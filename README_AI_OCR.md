# AI OCR Bank Statement Extractor

An enhanced Python script that uses multiple AI OCR engines to extract transaction data from bank statement PDFs and save it to CSV format.

## Features

- **Multiple OCR Engines**: Supports Tesseract, EasyOCR (deep learning), and Google Vision API
- **Automatic Method Selection**: Intelligently chooses the best OCR method for each document
- **Robust Text Extraction**: Falls back from direct PDF text extraction to OCR when needed
- **Smart Transaction Parsing**: Identifies dates, amounts, descriptions, and balances
- **CSV Output**: Structured data export with comprehensive transaction details
- **Detailed Reporting**: Generates summary reports with extraction statistics

## OCR Methods Comparison

| Method | Speed | Accuracy | Requirements | Best For |
|--------|-------|----------|--------------|----------|
| **Tesseract** | Fast | Good | System install | Clear, typed text |
| **EasyOCR** | Medium | Better | Python package | Handwritten text, various fonts |
| **Google Vision** | Slow | Best | Cloud API setup | Complex layouts, poor quality scans |
| **Auto** | Variable | Best | All above | Automatic selection for best results |

## Installation

### 1. System Dependencies

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

### 2. Python Dependencies

```bash
pip install -r requirements_ai_ocr.txt
```

### 3. Optional: Google Vision API Setup

1. Create a Google Cloud project and enable Vision API
2. Create a service account and download JSON key
3. Set environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/key.json"
```

## Quick Start

### Automated Setup
```bash
python setup_ai_ocr.py
```

### Manual Usage

1. **Place PDF files** in the `data_pdf` directory
2. **Run extraction**:
```bash
python ai_ocr_extractor.py --pdf-dir data_pdf --output results.csv
```

### Command Line Options

```bash
python ai_ocr_extractor.py [OPTIONS]

Options:
  --pdf-dir TEXT        Directory containing PDF files (default: data_pdf)
  --output TEXT         Output CSV file (default: ai_extracted_statements.csv)
  --ocr-method CHOICE   OCR method: auto, tesseract, easyocr, google_vision (default: auto)
  --report              Generate summary report
  --help                Show help message
```

## Usage Examples

### Basic Usage
```python
from ai_ocr_extractor import AIBankStatementExtractor

# Initialize extractor
extractor = AIBankStatementExtractor("data_pdf", "output.csv")

# Process all PDFs with automatic OCR selection
df = extractor.process_all_pdfs(ocr_method="auto")

# Generate report
report = extractor.generate_report(df)
```

### Specific OCR Method
```python
# Use only EasyOCR for better accuracy on handwritten text
df = extractor.process_all_pdfs(ocr_method="easyocr")
```

### Custom Processing
```python
# Extract text from a single PDF
text = extractor.extract_text_from_pdf("statement.pdf", "google_vision")

# Parse transactions from extracted text
transactions = extractor.parse_transactions(text, "statement.pdf")
```

## Output Format

The extracted data is saved as CSV with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Date | Transaction date | 2024-01-15 |
| Description | Transaction description | ATM WITHDRAWAL |
| Amount | Transaction amount | -100.00 |
| Balance | Account balance after transaction | 1500.00 |
| Type | Transaction type (if detected) | DEBIT |
| Source_File | Original PDF filename | statement_jan.pdf |

## Supported Bank Statement Formats

The extractor works with most standard bank statement formats including:

- **Date formats**: DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, DD MMM YYYY
- **Amount formats**: $1,234.56, 1234.56, (1234.56), ₹1,234.56
- **Layout types**: Tabular, list-based, multi-column

## Performance Tips

1. **Use 'auto' method** for best accuracy across different document types
2. **Preprocess PDFs** - ensure good quality scans (300+ DPI)
3. **Batch processing** - process multiple files together for efficiency
4. **GPU acceleration** - EasyOCR can use GPU for faster processing

## Troubleshooting

### Common Issues

**No text extracted:**
- Check if PDF is image-based (scanned)
- Verify OCR engines are properly installed
- Try different OCR methods

**Poor accuracy:**
- Use higher quality PDF scans
- Try Google Vision API for complex layouts
- Preprocess images (contrast, resolution)

**Missing transactions:**
- Check date/amount patterns in parse_transactions()
- Adjust confidence thresholds for EasyOCR
- Verify PDF contains readable text

### Error Messages

**"Tesseract not found":**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian  
sudo apt-get install tesseract-ocr
```

**"EasyOCR not available":**
```bash
pip install easyocr
```

**"Google Vision API failed":**
- Check API credentials and environment variable
- Verify API is enabled in Google Cloud Console
- Check internet connection

## Advanced Configuration

### Custom Transaction Patterns

Modify the `parse_transactions()` method to handle specific bank formats:

```python
# Add custom date pattern
date_patterns.append(r'\b(\d{2}-\w{3}-\d{4})\b')  # DD-MMM-YYYY

# Add custom amount pattern  
amount_patterns.append(r'INR\s*([0-9,]+\.?\d*)')  # INR amounts
```

### Image Preprocessing

Enhance OCR accuracy by customizing `preprocess_image()`:

```python
def preprocess_image(self, image):
    # Custom preprocessing for your specific documents
    # - Noise reduction
    # - Skew correction
    # - Binarization
    return processed_image
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues in the repository
3. Create a new issue with detailed information
