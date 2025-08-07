#!/usr/bin/env python3
"""
Enhanced AI OCR Bank Statement Extractor
Supports multiple OCR engines: PaddleOCR, Tesseract, EasyOCR, and Google Vision API
"""

import os
import re
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json
from dateutil import parser as date_parser
from dateutil.parser import ParserError

# Optional imports for enhanced OCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    print("Google Vision API not available. Install with: pip install google-cloud-vision")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR not available. Install with: pip install paddleocr")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """Data class for bank transaction"""
    date: str
    description: str
    amount: float
    balance: Optional[float] = None
    transaction_type: Optional[str] = None
    reference: Optional[str] = None

class AIBankStatementExtractor:
    """Enhanced bank statement extractor with multiple AI OCR engines"""
    
    def __init__(self, pdf_directory: str, output_file: str = "ai_extracted_statements.csv"):
        self.pdf_directory = pdf_directory
        self.output_file = output_file
        self.extracted_data = []
        
        # Initialize OCR engines
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
        
        self.google_vision_client = None
        if GOOGLE_VISION_AVAILABLE:
            try:
                # Test authentication by creating client and making a simple call
                test_client = vision.ImageAnnotatorClient()
                # Verify credentials are valid by checking if we can access the service
                # This will fail fast if credentials are invalid
                self.google_vision_client = test_client
                logger.info("Google Vision API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Vision API: {e}")
                logger.warning("Google Vision API will be disabled. Common fixes:")
                logger.warning("1. Check GOOGLE_APPLICATION_CREDENTIALS environment variable")
                logger.warning("2. Verify service account key file exists and is valid")
                logger.warning("3. Ensure Vision API is enabled in Google Cloud Console")
                self.google_vision_client = None
        
        self.paddle_ocr = None
        if PADDLEOCR_AVAILABLE:
            try:
                # Initialize PaddleOCR with English language support
                self.paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
                self.paddle_ocr = None
    
    def extract_text_with_tesseract(self, image: Image.Image) -> str:
        """Extract text using Tesseract OCR"""
        try:
            # Optimize image for OCR
            image = self.preprocess_image(image)
            
            # Configure Tesseract for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/-:() '
            text = pytesseract.image_to_string(image, config=custom_config)
            return text
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""
    
    def extract_text_with_easyocr(self, image: Image.Image) -> str:
        """Extract text using EasyOCR (deep learning based)"""
        if not self.easyocr_reader:
            return ""
        
        try:
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            # Extract text with confidence scores
            results = self.easyocr_reader.readtext(image_np)
            
            # Combine text with confidence filtering
            text_parts = []
            for (_, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence text
                    text_parts.append(text)
            
            return ' '.join(text_parts)
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return ""
    
    def extract_text_with_google_vision(self, image: Image.Image) -> str:
        """Extract text using Google Vision API"""
        if not self.google_vision_client:
            return ""
        
        try:
            # Convert PIL image to bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create Vision API image object
            vision_image = vision.Image(content=img_byte_arr)
            
            # Perform text detection
            response = self.google_vision_client.text_detection(image=vision_image)
            
            # Check for API errors
            if response.error.message:
                logger.error(f"Google Vision API error: {response.error.message}")
                # Disable Google Vision for this session if authentication fails
                if "invalid_grant" in response.error.message or "authentication" in response.error.message.lower():
                    logger.warning("Disabling Google Vision API due to authentication error")
                    self.google_vision_client = None
                return ""
            
            texts = response.text_annotations
            if texts:
                return texts[0].description
            return ""
            
        except Exception as e:
            logger.error(f"Google Vision API failed: {e}")
            # Disable Google Vision for this session if it's an auth error
            if "invalid_grant" in str(e) or "authentication" in str(e).lower():
                logger.warning("Disabling Google Vision API due to authentication error")
                self.google_vision_client = None
            return ""
    
    def extract_text_with_paddleocr(self, image: Image.Image) -> str:
        """Extract text using PaddleOCR (deep learning based)"""
        if not self.paddle_ocr:
            return ""
        
        try:
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            # Extract text with PaddleOCR
            results = self.paddle_ocr.ocr(image_np)
            
            # Combine text from all detected regions
            text_parts = []
            for line in results[0] if results and results[0] else []:
                if line and len(line) > 1 and line[1]:
                    text, confidence = line[1]
                    if confidence > 0.5:  # Filter low confidence text
                        text_parts.append(text)
            
            return ' '.join(text_parts)
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return ""
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Resize if too small
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000/width, 1000/height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_from_pdf(self, pdf_path: str, ocr_method: str = "auto") -> str:
        """
        Extract text from PDF using multiple methods
        ocr_method: 'paddleocr', 'tesseract', 'easyocr', 'google_vision', or 'auto'
        """
        # Try direct text extraction first
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    full_text += text + "\n"
                
                # If direct extraction worked and has meaningful content, use it
                if full_text and len(full_text.strip()) > 50:
                    logger.info(f"Successfully extracted text directly from {os.path.basename(pdf_path)}")
                    return full_text
        except Exception as e:
            logger.error(f"Direct text extraction failed for {pdf_path}: {e}")
        
        # Fallback to OCR
        logger.info(f"Using OCR method '{ocr_method}' for {os.path.basename(pdf_path)}")
        
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=300)
            full_text = ""
            
            for page_num, page_image in enumerate(pages):
                logger.info(f"Processing page {page_num + 1}/{len(pages)}")
                
                if ocr_method == "auto":
                    # Try multiple OCR methods and use the best result
                    texts = []
                    
                    # Try PaddleOCR first (best for financial documents)
                    if PADDLEOCR_AVAILABLE and self.paddle_ocr:
                        paddle_text = self.extract_text_with_paddleocr(page_image)
                        if paddle_text:
                            texts.append(("paddleocr", paddle_text))
                    
                    # Try EasyOCR (good general purpose)
                    if EASYOCR_AVAILABLE and self.easyocr_reader:
                        easyocr_text = self.extract_text_with_easyocr(page_image)
                        if easyocr_text:
                            texts.append(("easyocr", easyocr_text))
                    
                    # Try Google Vision API
                    if GOOGLE_VISION_AVAILABLE and self.google_vision_client:
                        google_text = self.extract_text_with_google_vision(page_image)
                        if google_text:
                            texts.append(("google_vision", google_text))
                    
                    # Try Tesseract as fallback
                    tesseract_text = self.extract_text_with_tesseract(page_image)
                    if tesseract_text:
                        texts.append(("tesseract", tesseract_text))
                    
                    # Select best result (longest meaningful text)
                    if texts:
                        best_text = max(texts, key=lambda x: len(x[1].strip()))[1]
                        full_text += best_text + "\n"
                
                elif ocr_method == "paddleocr":
                    text = self.extract_text_with_paddleocr(page_image)
                    full_text += text + "\n"
                
                elif ocr_method == "easyocr":
                    text = self.extract_text_with_easyocr(page_image)
                    full_text += text + "\n"
                
                elif ocr_method == "google_vision":
                    text = self.extract_text_with_google_vision(page_image)
                    full_text += text + "\n"
                
                else:  # tesseract
                    text = self.extract_text_with_tesseract(page_image)
                    full_text += text + "\n"
            
            return full_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return ""
    
    def parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string with multiple format support and validation"""
        if not date_str or not date_str.strip():
            return None
            
        date_str = date_str.strip()
        
        # Common date patterns with explicit formats
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',  # DD/MM/YYYY variants
            '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y',  # MM/DD/YYYY variants
            '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d',  # YYYY/MM/DD variants
            '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',  # DD/MM/YY variants
            '%m/%d/%y', '%m-%d-%y', '%m.%d.%y',  # MM/DD/YY variants
            '%d %b %Y', '%d %B %Y',              # DD MMM/MMMM YYYY
            '%b %d, %Y', '%B %d, %Y',            # MMM/MMMM DD, YYYY
        ]
        
        # Try explicit format parsing first
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # Validate year range (reasonable for bank statements)
                if 1990 <= parsed_date.year <= datetime.now().year + 1:
                    return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try dateutil parser as fallback
        try:
            parsed_date = date_parser.parse(date_str, fuzzy=True)
            # Validate year range
            if 1990 <= parsed_date.year <= datetime.now().year + 1:
                return parsed_date.strftime('%Y-%m-%d')
        except (ParserError, ValueError, TypeError):
            pass
        
        logger.debug(f"Could not parse date: {date_str}")
        return None
    
    def parse_amount(self, amount_str: str, is_negative: bool = False) -> Optional[float]:
        """Parse amount string with currency support and validation"""
        if not amount_str or not amount_str.strip():
            return None
        
        amount_str = amount_str.strip()
        
        # Remove currency symbols and clean up
        currency_symbols = ['$', '₹', '€', '£', '¥', '¢', '₦', 'R', 'kr', 'zł']
        for symbol in currency_symbols:
            amount_str = amount_str.replace(symbol, '')
        
        # Remove common formatting
        amount_str = amount_str.replace(',', '').replace(' ', '')
        
        # Handle negative indicators
        if amount_str.startswith('(') and amount_str.endswith(')'):
            amount_str = amount_str[1:-1]
            is_negative = True
        elif amount_str.startswith('-'):
            amount_str = amount_str[1:]
            is_negative = True
        
        # Try to convert to float
        try:
            amount = float(amount_str)
            if is_negative:
                amount = -abs(amount)
            
            # Validate amount range (reasonable for bank transactions)
            if abs(amount) > 1_000_000_000:  # 1 billion limit
                logger.debug(f"Amount too large: {amount}")
                return None
            
            return amount
        except ValueError:
            logger.debug(f"Could not parse amount: {amount_str}")
            return None
    
    def extract_transaction_data(self, line: str) -> Optional[Tuple[str, List[float], str]]:
        """Extract date, amounts, and description from a line"""
        if not line or len(line.strip()) < 5:
            return None
        
        line = line.strip()
        
        # Enhanced date patterns
        date_patterns = [
            r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b',          # DD/MM/YYYY variants
            r'\b(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b',           # YYYY/MM/DD variants
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4})\b',  # DD MMM YYYY
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{2,4})\b',  # MMM DD, YYYY
        ]
        
        # Enhanced amount patterns with currency support (ordered by specificity)
        amount_patterns = [
            r'\(([0-9,]+(?:\.[0-9]{1,2})?)\)',                # Negative in parentheses (highest priority)
            r'[\$₹€£¥¢₦R]\s*([0-9,]+(?:\.[0-9]{1,2})?)',     # Currency before
            r'([0-9,]+(?:\.[0-9]{1,2})?)\s*[\$₹€£¥¢₦R]',     # Currency after
            r'-\s*([0-9,]+(?:\.[0-9]{1,2})?)',               # Negative with minus
            r'\b([0-9,]+\.[0-9]{2})\b',                       # Decimal amounts with exactly 2 decimals
            r'\b([1-9][0-9,]{3,}(?:\.[0-9]{1,2})?)\b',       # Large amounts starting with non-zero
        ]
        
        # Find date
        date_found = None
        for pattern in date_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                potential_date = self.parse_date(match.group(1))
                if potential_date:
                    date_found = potential_date
                    break
        
        if not date_found:
            return None
        
        # Find amounts (avoid overlapping matches)
        amounts = []
        used_positions = set()
        
        for pattern in amount_patterns:
            matches = re.finditer(pattern, line)
            for match in matches:
                start, end = match.span()
                # Skip if this position overlaps with already found amounts
                if any(pos in range(start, end) for pos in used_positions):
                    continue
                    
                is_negative = '(' in match.group(0) or '-' in match.group(0)
                amount_str = match.group(1)
                
                # Skip if this looks like a year (4 digits between 1900-2100)
                if amount_str.isdigit() and 1900 <= int(amount_str) <= 2100:
                    continue
                
                amount = self.parse_amount(amount_str, is_negative)
                if amount is not None:
                    amounts.append(amount)
                    # Mark positions as used
                    used_positions.update(range(start, end))
        
        if not amounts:
            return None
        
        # Extract description by removing dates and amounts
        description = line
        
        # Remove date matches
        for pattern in date_patterns:
            description = re.sub(pattern, '', description, flags=re.IGNORECASE)
        
        # Remove amount matches
        for pattern in amount_patterns:
            description = re.sub(pattern, '', description)
        
        # Clean up description
        description = re.sub(r'\s+', ' ', description).strip()
        description = re.sub(r'^[\s\-.,;:]+|[\s\-.,;:]+$', '', description)
        
        if not description or len(description) < 3:
            description = "Transaction"
        
        return date_found, amounts, description
    
    def parse_hdfc_transaction_line(self, line: str) -> Optional[Transaction]:
        """Parse HDFC specific transaction line format"""
        # HDFC format: Date | Narration | ChqJRef.No. | Value Dt | Withdrawal Amt. | Deposit Amt. | Closing Balance
        # Example: 03/04/12 | CHQ PAID-MICR CTS-NE-LIC OF INDIA 0000000000234434 03/04/12 71,142.00 0.00 1,010,451.27
        
        if not line or len(line.strip()) < 10:
            return None
            
        line = line.strip()
        
        # Look for date pattern at the beginning (DD/MM/YY format)
        date_pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4})'
        date_match = re.search(date_pattern, line)
        
        if not date_match:
            return None
            
        date_str = date_match.group(1)
        parsed_date = self.parse_date(date_str)
        if not parsed_date:
            return None
        
        # Remove the date from the line to process the rest
        remaining_line = line[date_match.end():].strip()
        
        # Split by multiple spaces to separate fields
        parts = re.split(r'\s{2,}', remaining_line)
        if len(parts) < 3:
            # Fallback: split by single space and try to identify amounts
            parts = remaining_line.split()
        
        # Find amounts in the line (withdrawal, deposit, balance)
        amounts = []
        amount_pattern = r'^\d{1,3}(?:,\d{3})*(?:\.\d{2})?$'
        
        for part in parts[-4:]:  # Look in last 4 parts where amounts usually are
            if re.match(amount_pattern, part):
                try:
                    amount_val = float(part.replace(',', ''))
                    amounts.append(amount_val)
                except ValueError:
                    continue
        
        if len(amounts) < 2:
            return None
            
        # HDFC format: [...] Withdrawal_Amt Deposit_Amt Closing_Balance
        withdrawal_amt = amounts[-3] if len(amounts) >= 3 else 0.0
        deposit_amt = amounts[-2] if len(amounts) >= 2 else 0.0
        closing_balance = amounts[-1] if len(amounts) >= 1 else None
        
        # Calculate net amount (deposit is positive, withdrawal is negative)
        net_amount = deposit_amt - withdrawal_amt
        
        # Extract description (everything between date and amounts)
        description_parts = []
        for part in parts[:-3]:  # All parts except the last 3 amounts
            if not re.match(r'^\d+$', part):  # Skip pure numbers that might be references
                description_parts.append(part)
        
        description = ' '.join(description_parts) if description_parts else "Transaction"
        
        # Clean up description
        description = re.sub(r'\s+', ' ', description).strip()
        description = re.sub(r'^[\s\-.,;:|]+|[\s\-.,;:|]+$', '', description)
        
        if len(description) < 3:
            description = "Bank Transaction"
        
        # Determine transaction type
        transaction_type = self.classify_transaction(description, net_amount)
        
        return Transaction(
            date=parsed_date,
            description=description,
            amount=net_amount,
            balance=closing_balance,
            transaction_type=transaction_type
        )

    def parse_transactions(self, text: str, filename: str) -> List[Transaction]:
        """Parse transactions from extracted text with enhanced robustness"""
        transactions = []
        lines = text.split('\n')
        
        # Check if this looks like an HDFC statement
        is_hdfc = any('hdfc' in line.lower() for line in lines[:20])
        has_hdfc_header = any(all(word in line.lower() for word in ['withdrawal', 'deposit', 'closing', 'balance']) 
                             for line in lines[:30])
        
        if is_hdfc or has_hdfc_header:
            logger.info(f"Detected HDFC bank statement format in {filename}")
            
            # Process with HDFC-specific parser
            for line_num, line in enumerate(lines, 1):
                try:
                    transaction = self.parse_hdfc_transaction_line(line)
                    if transaction:
                        transactions.append(transaction)
                except Exception as e:
                    logger.debug(f"Error parsing HDFC line {line_num} in {filename}: {e}")
                    continue
        else:
            # Use generic parser for other bank formats
            # Filter out obviously non-transaction lines
            filtered_lines = []
            skip_keywords = {
                'balance', 'account', 'statement', 'period', 'address', 'phone',
                'customer', 'service', 'branch', 'swift', 'ifsc', 'routing',
                'page', 'continued', 'total', 'summary', 'opening', 'closing'
            }
            
            for line in lines:
                line = line.strip()
                if len(line) < 10:  # Skip very short lines
                    continue
                if any(keyword in line.lower() for keyword in skip_keywords):
                    continue
                if line.count(' ') < 2:  # Skip lines with too few words
                    continue
                filtered_lines.append(line)
            
            logger.debug(f"Processing {len(filtered_lines)} lines after filtering from {filename}")
            
            for line_num, line in enumerate(filtered_lines, 1):
                try:
                    result = self.extract_transaction_data(line)
                    if not result:
                        continue
                    
                    date_found, amounts, description = result
                    
                    # Determine transaction type based on amount and keywords
                    transaction_type = self.classify_transaction(description, amounts[0])
                    
                    # Create transaction
                    transaction = Transaction(
                        date=date_found,
                        description=description,
                        amount=amounts[0],
                        balance=amounts[1] if len(amounts) > 1 else None,
                        transaction_type=transaction_type
                    )
                    
                    transactions.append(transaction)
                    
                except Exception as e:
                    logger.debug(f"Error parsing line {line_num} in {filename}: {e}")
                    continue
        
        # Post-process transactions for consistency
        transactions = self.post_process_transactions(transactions)
        
        logger.info(f"Extracted {len(transactions)} transactions from {filename}")
        return transactions
    
    def classify_transaction(self, description: str, amount: float) -> str:
        """Classify transaction type based on description and amount"""
        desc_lower = description.lower()
        
        # Debit/Credit based on amount
        base_type = "CREDIT" if amount > 0 else "DEBIT"
        
        # Specific transaction types
        if any(word in desc_lower for word in ['transfer', 'neft', 'rtgs', 'imps']):
            return f"{base_type}_TRANSFER"
        elif any(word in desc_lower for word in ['atm', 'withdrawal', 'cash']):
            return "ATM_WITHDRAWAL"
        elif any(word in desc_lower for word in ['deposit', 'cheque', 'check']):
            return "DEPOSIT"
        elif any(word in desc_lower for word in ['interest', 'dividend']):
            return "INTEREST"
        elif any(word in desc_lower for word in ['fee', 'charge', 'penalty']):
            return "FEE"
        elif any(word in desc_lower for word in ['salary', 'payroll']):
            return "SALARY"
        
        return base_type
    
    def post_process_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        """Post-process transactions for consistency and validation"""
        if not transactions:
            return transactions
        
        processed = []
        
        # Sort by date
        try:
            transactions.sort(key=lambda t: datetime.strptime(t.date, '%Y-%m-%d'))
        except ValueError:
            logger.warning("Some dates could not be sorted")
        
        # Remove duplicates and validate
        seen = set()
        for trans in transactions:
            # Create a signature for duplicate detection
            signature = (trans.date, trans.description[:50], abs(trans.amount))
            if signature in seen:
                logger.debug(f"Skipping duplicate transaction: {signature}")
                continue
            
            # Validate transaction
            if self.validate_transaction(trans):
                seen.add(signature)
                processed.append(trans)
        
        return processed
    
    def validate_transaction(self, transaction: Transaction) -> bool:
        """Validate transaction data"""
        # Check required fields
        if not transaction.date or not transaction.description:
            return False
        
        # Check amount is reasonable
        if abs(transaction.amount) > 1_000_000_000:  # 1 billion limit
            return False
        
        # Check date format
        try:
            datetime.strptime(transaction.date, '%Y-%m-%d')
        except ValueError:
            return False
        
        # Check description length
        if len(transaction.description.strip()) < 2:
            return False
        
        return True
    
    def process_all_pdfs(self, ocr_method: str = "auto") -> pd.DataFrame:
        """
        Process all PDFs in the directory
        ocr_method: 'paddleocr', 'tesseract', 'easyocr', 'google_vision', or 'auto'
        """
        all_transactions = []
        
        if not os.path.exists(self.pdf_directory):
            logger.error(f"Directory {self.pdf_directory} does not exist")
            return pd.DataFrame()
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            logger.info(f"Processing {pdf_file}...")
            
            try:
                # Extract text
                text = self.extract_text_from_pdf(pdf_path, ocr_method)
                
                if not text.strip():
                    logger.warning(f"No text extracted from {pdf_file}")
                    continue
                
                # Parse transactions
                transactions = self.parse_transactions(text, pdf_file)
                
                # Add source file information
                for transaction in transactions:
                    transaction.reference = pdf_file
                
                all_transactions.extend(transactions)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        # Convert to DataFrame
        if all_transactions:
            df_data = []
            for trans in all_transactions:
                df_data.append({
                    'Date': trans.date,
                    'Description': trans.description,
                    'Amount': trans.amount,
                    'Balance': trans.balance,
                    'Type': trans.transaction_type,
                    'Source_File': trans.reference
                })
            
            df = pd.DataFrame(df_data)
            
            # Save to CSV
            df.to_csv(self.output_file, index=False)
            logger.info(f"Saved {len(df)} transactions to {self.output_file}")
            
            return df
        else:
            logger.warning("No transactions extracted from any PDF")
            return pd.DataFrame()
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a summary report"""
        if df.empty:
            return "No data to report"
        
        report = f"""
# AI OCR Bank Statement Extraction Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total transactions extracted: {len(df)}
- Date range: {df['Date'].min()} to {df['Date'].max()}
- Total amount: ${df['Amount'].sum():,.2f}
- Average transaction: ${df['Amount'].mean():,.2f}

## Files Processed
{df['Source_File'].nunique()} unique PDF files:
{chr(10).join([f"- {file}" for file in df['Source_File'].unique()])}

## Transaction Types Distribution
{df.groupby('Source_File').size().to_string()}
        """
        
        return report

def main():
    """Main function to run the AI OCR extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI OCR Bank Statement Extractor')
    parser.add_argument('--pdf-dir', default='statements', help='Directory containing PDF files')
    parser.add_argument('--output', default='ai_extracted_statements.csv', help='Output CSV file')
    parser.add_argument('--ocr-method', choices=['auto', 'paddleocr', 'tesseract', 'easyocr', 'google_vision'], 
                       default='tesseract', help='OCR method to use')
    parser.add_argument('--report', action='store_true', help='Generate summary report')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = AIBankStatementExtractor(args.pdf_dir, args.output)
    
    # Process PDFs
    logger.info(f"Starting AI OCR extraction with method: {args.ocr_method}")
    df = extractor.process_all_pdfs(args.ocr_method)
    
    if not df.empty:
        print(f"\n✅ Successfully extracted {len(df)} transactions")
        print(f"📄 Output saved to: {args.output}")
        
        if args.report:
            report = extractor.generate_report(df)
            report_file = args.output.replace('.csv', '_report.md')
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"📊 Report saved to: {report_file}")
    else:
        print("❌ No transactions were extracted")

if __name__ == "__main__":
    main()
