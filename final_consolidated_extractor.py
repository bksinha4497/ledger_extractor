import os
import re
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2
import logging
from datetime import datetime
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalConsolidatedPDFLedgerExtractor:
    def __init__(self, pdf_directory: str, output_file: str = "final_consolidated_output.csv"):
        self.pdf_directory = pdf_directory
        self.output_file = output_file
        self.extracted_data = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using direct extraction first, then OCR as fallback"""
        # Try direct text extraction first
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    logger.info(f"Extracting text from page {page_num + 1} of {os.path.basename(pdf_path)}")
                    text = page.extract_text()
                    full_text += text + "\n"
                
                # If direct extraction worked and has meaningful content, use it
                if full_text and len(full_text.strip()) > 50:
                    logger.info(f"Successfully extracted text directly from {os.path.basename(pdf_path)}")
                    return full_text
        except Exception as e:
            logger.error(f"Error extracting text directly from {pdf_path}: {str(e)}")
        
        # Fallback to OCR
        logger.info(f"Direct extraction failed or insufficient, trying OCR for {os.path.basename(pdf_path)}")
        try:
            pages = convert_from_path(pdf_path, dpi=300)
            full_text = ""
            
            for page_num, page in enumerate(pages):
                logger.info(f"OCR processing page {page_num + 1} of {os.path.basename(pdf_path)}")
                text = pytesseract.image_to_string(page, config='--psm 6')
                full_text += text + "\n"
                
            return full_text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def parse_date(self, date_str: str) -> str:
        """Parse and standardize date format"""
        return date_str.strip()
    
    def is_valid_transaction_date(self, date_str: str, context_line: str) -> bool:
        """Check if a date string represents a valid transaction date"""
        
        # Skip date ranges (financial year headers)
        if 'to' in context_line.lower() and len(context_line) < 50:
            return False
        
        # Skip if it's part of header information
        header_keywords = ['ledger account', 'page', 'bangalore', 'sanjay kumar', 'particulars', 'credit', 'debit']
        if any(keyword in context_line.lower() for keyword in header_keywords):
            return False
        
        # Skip if it looks like a date range pattern
        if re.search(r'\d{1,2}-[A-Za-z]{3}-\d{2}\s+to\s+\d{1,2}-[A-Za-z]{3}-\d{2}', context_line):
            return False
        
        return True
    
    def extract_ledger_entries(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Extract ledger entries using comprehensive parsing logic"""
        entries = []
        
        # Split text into lines for better parsing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Find all dates and their associated transactions
        date_pattern = r'\d{1,2}-[A-Za-z]{3}-\d{2}'
        
        i = 0
        current_date = ""
        
        while i < len(lines):
            line = lines[i]
            
            # Look for date in current line
            date_match = re.search(date_pattern, line)
            if date_match and self.is_valid_transaction_date(date_match.group(0), line):
                current_date = date_match.group(0)
                
                # Collect all lines that might be part of this transaction
                transaction_lines = [line]
                j = i + 1
                
                # Look ahead for related transaction lines
                while j < len(lines) and j < i + 15:  # Look ahead up to 15 lines
                    next_line = lines[j]
                    
                    # Stop if we hit another date (unless it's the same date)
                    next_date_match = re.search(date_pattern, next_line)
                    if next_date_match and self.is_valid_transaction_date(next_date_match.group(0), next_line):
                        if next_date_match.group(0) != current_date:
                            break
                    
                    # Stop at certain keywords that indicate end of transaction
                    if re.search(r'^(Carried Over|Brought Forward|continued|Page \d+)', next_line, re.IGNORECASE):
                        break
                    
                    transaction_lines.append(next_line)
                    j += 1
                
                # Parse the collected transaction block
                block_text = '\n'.join(transaction_lines)
                parsed_entries = self.parse_consolidated_transaction_block(block_text, current_date, filename)
                entries.extend(parsed_entries)
                
                i = j  # Move to next unprocessed line
            else:
                # Check if this line contains transaction info without date (use current_date)
                if current_date and self.has_transaction_info(line):
                    # Collect transaction lines without date
                    transaction_lines = [line]
                    j = i + 1
                    
                    # Look ahead for related transaction lines
                    while j < len(lines) and j < i + 10:
                        next_line = lines[j]
                        
                        # Stop if we hit a date
                        if re.search(date_pattern, next_line):
                            break
                        
                        # Stop at certain keywords
                        if re.search(r'^(Carried Over|Brought Forward|continued|Page \d+)', next_line, re.IGNORECASE):
                            break
                        
                        transaction_lines.append(next_line)
                        j += 1
                    
                    # Parse the collected transaction block with current_date
                    block_text = '\n'.join(transaction_lines)
                    parsed_entries = self.parse_consolidated_transaction_block(block_text, current_date, filename)
                    entries.extend(parsed_entries)
                    
                    i = j
                else:
                    i += 1
        
        # Consolidate entries by Voucher Number
        consolidated_entries = self.consolidate_by_voucher_number(entries)
        
        logger.info(f"Extracted {len(consolidated_entries)} consolidated entries from {filename}")
        return consolidated_entries
    
    def has_transaction_info(self, line: str) -> bool:
        """Check if line contains transaction information"""
        # First check if it's an opening/closing balance - exclude these
        if self.is_balance_entry(line):
            return False
            
        # Look for patterns that indicate transaction data
        patterns = [
            r'(Cr|Dr)\s+',
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # Amount pattern
            r'\d{2,4}\s+(Payment|Receipt|Purchase)',  # Voucher number + type
            r'(Payment|Receipt|Purchase)',
            r'HDFC Bank',
            r'(CGST|SGST|IGST)',
            r'PURCHASES'
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)
    
    def is_balance_entry(self, text: str) -> bool:
        """Check if the text represents an opening or closing balance entry"""
        # Only check if the text is specifically a balance entry line
        # Make it more specific to avoid false positives
        balance_patterns = [
            r'^\s*\d{1,2}-[A-Za-z]{3}-\d{2}\s+Dr\s+Opening\s+Balance',
            r'^\s*\d{1,2}-[A-Za-z]{3}-\d{2}\s+Cr\s+Opening\s+Balance',
            r'^\s*\d{1,2}-[A-Za-z]{3}-\d{2}\s+Dr\s+Closing\s+Balance',
            r'^\s*\d{1,2}-[A-Za-z]{3}-\d{2}\s+Cr\s+Closing\s+Balance',
            r'Opening\s+Balance.*?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$',
            r'Closing\s+Balance.*?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$'
        ]
        
        # Check if the entire text block is just a balance entry
        lines = text.strip().split('\n')
        if len(lines) <= 3:  # Only check short text blocks
            for pattern in balance_patterns:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    return True
        
        return False
    
    def parse_consolidated_transaction_block(self, block_text: str, date_str: str, filename: str) -> List[Dict[str, Any]]:
        """Parse a transaction block and return consolidated entries by voucher number"""
        entries = []
        
        # Skip if this is a balance entry (Opening/Closing Balance)
        if self.is_balance_entry(block_text):
            return entries  # Return empty list to skip balance entries
        
        # Extract all voucher numbers from the block
        voucher_numbers = self.extract_all_voucher_numbers(block_text)
        
        if not voucher_numbers:
            # If no voucher numbers found, try to extract a single transaction
            entry = self.extract_single_transaction(block_text, date_str, filename)
            if entry:
                entries.append(entry)
        else:
            # Process each voucher number separately
            for vch_no in voucher_numbers:
                entry = self.extract_transaction_by_voucher(block_text, date_str, filename, vch_no)
                if entry:
                    entries.append(entry)
        
        return entries
    
    def extract_all_voucher_numbers(self, block_text: str) -> List[str]:
        """Extract all voucher numbers from block text"""
        voucher_numbers = []
        
        # Pattern to find voucher numbers - more specific patterns first
        vch_patterns = [
            r'Vch No[.\s]*(\d+)',  # Explicit voucher number
            r'(\d{2,4})\s+(?:Payment|Receipt|Purchase)',  # Number followed by transaction type
        ]
        
        for pattern in vch_patterns:
            matches = re.finditer(pattern, block_text, re.IGNORECASE)
            for match in matches:
                vch_no = match.group(1)
                if vch_no not in voucher_numbers and len(vch_no) >= 2:
                    voucher_numbers.append(vch_no)
        
        # Only if no voucher numbers found with specific patterns, try more general approach
        if not voucher_numbers:
            # Look for standalone numbers that are likely voucher numbers
            # But exclude numbers that are clearly part of bank details or other contexts
            lines = block_text.split('\n')
            for line in lines:
                line = line.strip()
                
                # Skip lines that contain bank details or account numbers
                if re.search(r'HDFC Bank|NEFT|IMPS|POS|XXXXXXXXXXX|MUM-N\d+', line, re.IGNORECASE):
                    continue
                
                # Skip lines that are clearly amounts (with decimal points or currency context)
                if re.search(r'\d+\.\d{2}|Cr\s+\d+|Dr\s+\d+', line):
                    continue
                
                # Look for standalone 3-4 digit numbers that could be voucher numbers
                standalone_numbers = re.findall(r'\b(\d{3,4})\b', line)
                for num in standalone_numbers:
                    if num not in voucher_numbers and len(num) >= 3:
                        # Additional validation - avoid common false positives
                        if not re.search(rf'{num}(?:\.\d{{2}}|\s*(?:Cr|Dr))', block_text, re.IGNORECASE):
                            voucher_numbers.append(num)
        
        return voucher_numbers
    
    def extract_transaction_by_voucher(self, block_text: str, date_str: str, filename: str, vch_no: str) -> Dict[str, Any]:
        """Extract a single consolidated transaction for a specific voucher number"""
        entry = self.create_entry_template(filename, date_str)
        entry['Vch No.'] = vch_no
        
        # Determine transaction type
        vch_type = self.determine_transaction_type(block_text, "")
        entry['Vch Type'] = vch_type
        
        # Analyze PDF structure (for now use default, can be enhanced later)
        pdf_structure = {'credit_position': 'right', 'debit_position': 'left'}
        
        # Extract amounts (Credit or Debit)
        credit_amount, debit_amount = self.extract_consolidated_amounts(block_text, vch_no, pdf_structure)
        entry['Credit'] = credit_amount
        entry['Debit'] = debit_amount
        
        # Build consolidated particulars
        particulars = self.build_consolidated_particulars(block_text, vch_no)
        entry['Particulars'] = particulars
        
        return entry
    
    def extract_single_transaction(self, block_text: str, date_str: str, filename: str) -> Dict[str, Any]:
        """Extract a single transaction when no voucher number is found"""
        entry = self.create_entry_template(filename, date_str)
        
        # Try to extract voucher number
        vch_no = self.extract_voucher_number(block_text)
        entry['Vch No.'] = vch_no
        
        # Determine transaction type
        vch_type = self.determine_transaction_type(block_text, "")
        entry['Vch Type'] = vch_type
        
        # Analyze PDF structure (for now use default, can be enhanced later)
        pdf_structure = {'credit_position': 'right', 'debit_position': 'left'}
        
        # Extract amounts
        credit_amount, debit_amount = self.extract_consolidated_amounts(block_text, vch_no, pdf_structure)
        entry['Credit'] = credit_amount
        entry['Debit'] = debit_amount
        
        # Build particulars
        particulars = self.build_consolidated_particulars(block_text, vch_no)
        entry['Particulars'] = particulars
        
        return entry
    
    def analyze_pdf_structure(self, text: str) -> Dict[str, str]:
        """Analyze PDF structure to determine Credit/Debit column positions"""
        lines = text.split('\n')
        
        # Look for header line that contains "Credit" and "Debit"
        header_info = {'credit_position': 'right', 'debit_position': 'left'}
        
        for line in lines:
            line = line.strip().upper()
            if 'CREDIT' in line and 'DEBIT' in line:
                # Find positions of Credit and Debit in the header
                credit_pos = line.find('CREDIT')
                debit_pos = line.find('DEBIT')
                
                if credit_pos < debit_pos:
                    header_info = {'credit_position': 'left', 'debit_position': 'right'}
                else:
                    header_info = {'credit_position': 'right', 'debit_position': 'left'}
                break
        
        return header_info
    
    def normalize_amount(self, amount_str: str) -> str:
        """Normalize amount string handling various currency formats"""
        if not amount_str:
            return ""
        
        # Remove any non-digit, non-comma, non-period characters
        cleaned = re.sub(r'[^\d,.]', '', str(amount_str))
        
        if not cleaned:
            return ""
        
        # Handle Indian currency format (1,57,000) and international format (157,000)
        # Remove all commas first, then check if it's a valid number
        no_commas = cleaned.replace(',', '')
        
        # Check if it's a valid number
        try:
            float(no_commas)
            return no_commas
        except ValueError:
            # If not valid, try to extract the largest continuous number
            numbers = re.findall(r'\d+', cleaned)
            if numbers:
                # Join all number parts
                return ''.join(numbers)
            return ""
    
    def extract_consolidated_amounts(self, block_text: str, vch_no: str, pdf_structure: Dict[str, str]) -> tuple:
        """Extract consolidated credit and debit amounts for a specific voucher number"""
        credit_amount = ""
        debit_amount = ""
        
        # Split block into lines to process each transaction separately
        lines = block_text.split('\n')
        
        # Look for the specific voucher number and its associated amount
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if this line contains our voucher number
            if vch_no in line:
                # Look for Cr or Dr indicator in this line or nearby lines
                context_lines = lines[max(0, i-3):i+4]  # Get context around the voucher line
                context_text = '\n'.join(context_lines)
                
                # Enhanced amount patterns to capture various formats
                amount_patterns = [
                    r'(\d{1,2},\d{2},\d{3}(?:\.\d{2})?)',  # Indian format: 1,57,000.00
                    r'(\d{1,3},\d{3},\d{3}(?:\.\d{2})?)',  # International format: 157,000.00
                    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',   # General comma format
                    r'(\d+(?:\.\d{2})?)'                   # Simple number format
                ]
                
                # Check for Credit transaction with this voucher number
                if 'Cr' in context_text or 'Credit' in context_text:
                    for amount_pattern in amount_patterns:
                        cr_patterns = [
                            rf'Cr\s+.*?{amount_pattern}\s+{vch_no}',
                            rf'{vch_no}.*?{amount_pattern}.*?Payment',
                            rf'{amount_pattern}\s+{vch_no}.*?Payment',
                            rf'Cr.*?{amount_pattern}.*?{vch_no}',
                        ]
                        
                        for pattern in cr_patterns:
                            match = re.search(pattern, context_text, re.IGNORECASE)
                            if match:
                                raw_amount = match.group(1)
                                credit_amount = self.normalize_amount(raw_amount)
                                if credit_amount:
                                    return credit_amount, debit_amount
                
                # Check for Debit transaction with this voucher number
                if 'Dr' in context_text or 'Debit' in context_text:
                    for amount_pattern in amount_patterns:
                        dr_patterns = [
                            rf'Dr\s+.*?{amount_pattern}\s+{vch_no}',
                            rf'{vch_no}.*?{amount_pattern}.*?Purchase',
                            rf'{amount_pattern}\s+{vch_no}.*?Purchase',
                            rf'Dr.*?{amount_pattern}.*?{vch_no}',
                        ]
                        
                        for pattern in dr_patterns:
                            match = re.search(pattern, context_text, re.IGNORECASE)
                            if match:
                                raw_amount = match.group(1)
                                debit_amount = self.normalize_amount(raw_amount)
                                if debit_amount:
                                    return credit_amount, debit_amount
        
        # Fallback: broader search with enhanced amount patterns
        for amount_pattern in [
            r'(\d{1,2},\d{2},\d{3}(?:\.\d{2})?)',  # Indian format
            r'(\d{1,3},\d{3},\d{3}(?:\.\d{2})?)',  # International format
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',   # General comma format
            r'(\d+(?:\.\d{2})?)'                   # Simple number format
        ]:
            # Look for Credit transactions
            cr_match = re.search(rf'Cr\s+[^0-9]*{amount_pattern}\s+{vch_no}', block_text, re.IGNORECASE)
            if cr_match:
                credit_amount = self.normalize_amount(cr_match.group(1))
                break
            
            # Look for Debit transactions  
            dr_match = re.search(rf'Dr\s+[^0-9]*{amount_pattern}\s+{vch_no}', block_text, re.IGNORECASE)
            if dr_match:
                debit_amount = self.normalize_amount(dr_match.group(1))
                break
        
        return credit_amount, debit_amount
    
    def build_consolidated_particulars(self, block_text: str, vch_no: str) -> str:
        """Build consolidated particulars for a transaction"""
        particulars_parts = []
        
        # Check for "(as per details)" pattern
        if re.search(r'\(as per details\)', block_text, re.IGNORECASE):
            particulars_parts.append("(as per details)")
            
            # Add GST breakdown
            gst_info = self.extract_gst_breakdown(block_text)
            if gst_info['types']:
                particulars_parts.extend(gst_info['types'])
            
            # Look for BEING statements
            being_match = re.search(r'BEING\s+([A-Z\s]+(?:VIDE|INVOICE|NO)[^0-9\n]*)', block_text, re.IGNORECASE)
            if being_match:
                being_text = being_match.group(1).strip()
                if len(being_text) < 50:
                    particulars_parts.append(f"BEING {being_text}")
        
        else:
            # Clean the block text by removing date and Cr/Dr indicators from the beginning
            cleaned_text = self.clean_particulars_text(block_text)
            
            # Look for bank details - include the complete bank line with numbers
            if 'HDFC Bank' in cleaned_text:
                # Find the complete HDFC Bank line including any numbers that are part of it
                bank_lines = []
                lines = cleaned_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if 'HDFC Bank' in line:
                        # Clean the bank line to remove date and Cr/Dr prefixes
                        clean_bank_line = self.clean_bank_line(line)
                        if clean_bank_line:
                            bank_lines.append(clean_bank_line)
                        break
                
                if bank_lines:
                    particulars_parts.extend(bank_lines)
            
            # Look for payment method details (NEFT, IMPS, POS etc.)
            payment_details = self.extract_payment_details(cleaned_text)
            particulars_parts.extend(payment_details[:3])  # Include more details
            
            # If no specific details found, use generic description
            if not particulars_parts:
                if re.search(r'Bank|HDFC|IMPS|POS|NEFT', cleaned_text, re.IGNORECASE):
                    particulars_parts.append("Bank Transaction")
                else:
                    particulars_parts.append("Transaction details")
        
        return '\n'.join(particulars_parts) if particulars_parts else "Transaction"
    
    def clean_particulars_text(self, text: str) -> str:
        """Clean text by removing date and Cr/Dr indicators from particulars"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Remove date pattern from the beginning of line
            line = re.sub(r'^\d{1,2}-[A-Za-z]{3}-\d{2}\s+', '', line)
            # Remove Cr/Dr indicators from the beginning
            line = re.sub(r'^(Cr|Dr)\s*', '', line)
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def clean_bank_line(self, line: str) -> str:
        """Clean bank line by removing date and Cr/Dr prefixes"""
        # Remove date pattern from the beginning
        line = re.sub(r'^\d{1,2}-[A-Za-z]{3}-\d{2}\s+', '', line)
        # Remove Cr/Dr indicators from the beginning
        line = re.sub(r'^(Cr|Dr)\s*', '', line)
        return line.strip()
    
    def consolidate_by_voucher_number(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate multiple entries with the same voucher number into single entries"""
        consolidated = {}
        
        for entry in entries:
            vch_no = entry.get('Vch No.', '')
            file_date_key = f"{entry['File Name']}_{entry['Date']}_{vch_no}"
            
            if file_date_key in consolidated:
                # Merge with existing entry
                existing = consolidated[file_date_key]
                
                # Keep the non-empty amounts
                if entry['Credit'] and not existing['Credit']:
                    existing['Credit'] = entry['Credit']
                if entry['Debit'] and not existing['Debit']:
                    existing['Debit'] = entry['Debit']
                
                # Merge particulars (avoid duplicates)
                existing_parts = set(existing['Particulars'].split('\n'))
                new_parts = set(entry['Particulars'].split('\n'))
                all_parts = existing_parts.union(new_parts)
                existing['Particulars'] = '\n'.join(sorted(all_parts))
                
            else:
                # Add new entry
                consolidated[file_date_key] = entry.copy()
        
        return list(consolidated.values())
    
    def extract_voucher_number(self, block_text: str) -> str:
        """Extract voucher number from block text"""
        vch_patterns = [
            r'Vch No[.\s]*(\d+)',  # Explicit voucher number
            r'(\d{1,4})\s+(?:Payment|Receipt|Purchase)',  # Number followed by transaction type
        ]
        
        for pattern in vch_patterns:
            match = re.search(pattern, block_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Only if no explicit voucher number found, try to find standalone numbers
        # but avoid numbers that are clearly part of bank details
        lines = block_text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip lines that contain bank details or account numbers
            if re.search(r'HDFC Bank|NEFT|IMPS|POS|XXXXXXXXXXX|MUM-N\d+', line, re.IGNORECASE):
                continue
            
            # Skip lines that are clearly amounts
            if re.search(r'\d+\.\d{2}|Cr\s+\d+|Dr\s+\d+', line):
                continue
            
            # Look for standalone 3-4 digit numbers
            standalone_numbers = re.findall(r'\b(\d{3,4})\b', line)
            for num in standalone_numbers:
                if len(num) >= 3:
                    # Additional validation - avoid common false positives
                    if not re.search(rf'{num}(?:\.\d{{2}}|\s*(?:Cr|Dr))', block_text, re.IGNORECASE):
                        return num
        
        return ""
    
    def determine_transaction_type(self, block_text: str, default_type: str) -> str:
        """Determine transaction type showing detailed breakdown as in PDF"""
        # Look for explicit voucher type in the text
        explicit_types = ['Payment', 'Purchase', 'Receipt', 'Journal', 'Contra']
        base_type = ""
        for vch_type in explicit_types:
            if re.search(rf'\b{vch_type}\b', block_text, re.IGNORECASE):
                base_type = vch_type
                break
        
        # If no explicit type found, determine from context
        if not base_type:
            if re.search(r'Bank|HDFC|IMPS|POS|NEFT|RTGS|UPI', block_text, re.IGNORECASE):
                base_type = "Payment"
            elif re.search(r'Purchase|GST|CGST|SGST|IGST|PURCHASES', block_text, re.IGNORECASE):
                base_type = "Purchase"
            elif re.search(r'Receipt', block_text, re.IGNORECASE):
                base_type = "Receipt"
            else:
                base_type = "Payment" if default_type == "Credit" else "Purchase"
        
        # For Purchase transactions, show the detailed breakdown as in PDF
        if base_type == "Purchase" and re.search(r'PURCHASES|CGST|SGST|IGST', block_text, re.IGNORECASE):
            # Extract the detailed breakdown from the PDF format
            breakdown_parts = []
            
            # Look for PURCHASES amount
            purchases_match = re.search(r'PURCHASES\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Dr', block_text, re.IGNORECASE)
            if purchases_match:
                amount = purchases_match.group(1).replace(',', '')
                breakdown_parts.append(f"PURCHASES\n{amount} Dr")
            else:
                breakdown_parts.append("PURCHASES")
            
            # Look for GST amounts in order
            gst_patterns = [
                (r'CGST\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Dr', 'CGST'),
                (r'SGST\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Dr', 'SGST'),
                (r'IGST\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Dr', 'IGST')
            ]
            
            for pattern, gst_type in gst_patterns:
                matches = re.finditer(pattern, block_text, re.IGNORECASE)
                for match in matches:
                    amount = match.group(1).replace(',', '')
                    breakdown_parts.append(f"{gst_type}\n{amount} Dr")
            
            # If we found detailed breakdown, return it
            if len(breakdown_parts) > 1:
                return '\n'.join(breakdown_parts)
        
        # For all transaction types, return only the base type (no payment method details)
        return base_type if base_type else "Transaction"
    
    def extract_payment_details(self, block_text: str) -> List[str]:
        """Extract payment method details"""
        details = []
        
        # Payment methods
        payment_methods = ['POS', 'NEFT', 'IMPS', 'UPI', 'RTGS']
        for method in payment_methods:
            method_match = re.search(f'{method}[^\\n]*', block_text, re.IGNORECASE)
            if method_match:
                detail = method_match.group(0).strip()
                if len(detail) < 100:
                    details.append(detail)
                break
        
        # Bank account details
        account_match = re.search(r'XXXXXXXXXXX\d+', block_text)
        if account_match:
            details.append(account_match.group(0))
        
        return details
    
    def extract_gst_breakdown(self, block_text: str) -> Dict[str, List[str]]:
        """Extract GST breakdown from transaction block"""
        gst_types = []
        gst_amounts = []
        
        # Look for PURCHASES first
        purchases_match = re.search(r'PURCHASES\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Dr', block_text, re.IGNORECASE)
        if purchases_match:
            gst_types.append('PURCHASES')
            gst_amounts.append(purchases_match.group(1).replace(',', ''))
        
        # Extract GST amounts in order
        gst_patterns = [
            (r'CGST\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Dr', 'CGST'),
            (r'SGST\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Dr', 'SGST'),
            (r'IGST\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Dr', 'IGST')
        ]
        
        for pattern, gst_type in gst_patterns:
            matches = re.finditer(pattern, block_text, re.IGNORECASE)
            for match in matches:
                gst_types.append(gst_type)
                gst_amounts.append(match.group(1).replace(',', ''))
        
        return {'types': gst_types, 'amounts': gst_amounts}
    
    def create_entry_template(self, filename: str, date_str: str) -> Dict[str, Any]:
        """Create a template entry dictionary"""
        return {
            'File Name': filename,
            'Date': self.parse_date(date_str),
            'Particulars': '',
            'Vch Type': '',
            'Vch No.': '',
            'Debit': '',
            'Credit': ''
        }
    
    def process_all_pdfs(self):
        """Process all PDF files in the directory"""
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.pdf')]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            logger.info(f"Processing: {pdf_file}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            if text:
                # Extract ledger entries
                entries = self.extract_ledger_entries(text, pdf_file)
                self.extracted_data.extend(entries)
            else:
                logger.warning(f"No text extracted from {pdf_file}")
    
    def save_to_csv(self):
        """Save extracted data to CSV file"""
        if not self.extracted_data:
            logger.warning("No data to save")
            return
        
        df = pd.DataFrame(self.extracted_data)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Data saved to {self.output_file}")
        
        # Print summary
        print(f"\nFinal Consolidated Extraction Summary:")
        print(f"Total entries extracted: {len(self.extracted_data)}")
        print(f"Files processed: {len(set([entry['File Name'] for entry in self.extracted_data]))}")
        print(f"Entries with Credit values: {len([e for e in self.extracted_data if e['Credit']])}")
        print(f"Entries with Debit values: {len([e for e in self.extracted_data if e['Debit']])}")
        print(f"Output saved to: {self.output_file}")
        
        # Show files with entry counts
        file_counts = {}
        for entry in self.extracted_data:
            file_name = entry['File Name']
            file_counts[file_name] = file_counts.get(file_name, 0) + 1
        
        print(f"\nFiles processed: {len(file_counts)} out of 90 total PDFs")
        print(f"Missing files: {90 - len(file_counts)}")
        
        print(f"\nTop 10 files by entry count:")
        sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        for file_name, count in sorted_files[:10]:
            print(f"  {file_name}: {count} entries")

def main():
    # Configuration
    PDF_DIRECTORY = "./data_pdf/"
    OUTPUT_FILE = "final_consolidated_output.csv"
    
    # Check if directory exists
    if not os.path.exists(PDF_DIRECTORY):
        print(f"Error: Directory '{PDF_DIRECTORY}' not found!")
        return
    
    # Initialize extractor
    extractor = FinalConsolidatedPDFLedgerExtractor(PDF_DIRECTORY, OUTPUT_FILE)
    
    try:
        # Process all PDFs
        extractor.process_all_pdfs()
        
        # Save results
        extractor.save_to_csv()
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
