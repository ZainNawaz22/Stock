#!/usr/bin/env python3
"""
Script to examine PDF structure
"""

import pdfplumber
import pandas as pd

def examine_pdf():
    # Open the PDF and examine its structure
    with pdfplumber.open('data/2025-07-25.pdf') as pdf:
        print(f'Number of pages: {len(pdf.pages)}')
        
        # Look at header structure
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        lines = text.split('\n')
        
        print('Header lines:')
        for i, line in enumerate(lines[:15]):
            print(f'{i}: {repr(line)}')
        
        # Find data lines (skip headers and section markers)
        data_lines = []
        for line in lines:
            # Skip empty lines, headers, and section markers
            if (line.strip() and 
                not line.startswith('***') and 
                not line.startswith('Pakistan Stock') and
                not line.startswith('CLOSING RATE') and
                not line.startswith('From :') and
                not line.startswith('PageNo:') and
                not line.startswith('Friday') and
                not line.startswith('Flu No:') and
                not line.startswith('P. Vol.:') and
                not line.startswith('C. Vol.:') and
                not line.startswith('Total:') and
                not line.startswith('Company Name')):
                
                # Check if line has numeric data (likely a stock entry)
                if any(char.isdigit() for char in line):
                    data_lines.append(line)
        
        print(f'\nFound {len(data_lines)} potential data lines on first page')
        print('First 5 data lines:')
        for i, line in enumerate(data_lines[:5]):
            print(f'{i}: {repr(line)}')
        
        # Check a few more pages
        total_data_lines = len(data_lines)
        for page_num in range(1, min(3, len(pdf.pages))):
            page = pdf.pages[page_num]
            page_text = page.extract_text()
            page_lines = page_text.split('\n')
            
            page_data_lines = []
            for line in page_lines:
                if (line.strip() and 
                    not line.startswith('***') and 
                    not line.startswith('Pakistan Stock') and
                    not line.startswith('CLOSING RATE') and
                    not line.startswith('From :') and
                    not line.startswith('PageNo:') and
                    not line.startswith('Friday') and
                    not line.startswith('Flu No:') and
                    not line.startswith('P. Vol.:') and
                    not line.startswith('C. Vol.:') and
                    not line.startswith('Total:') and
                    not line.startswith('Company Name')):
                    
                    if any(char.isdigit() for char in line):
                        page_data_lines.append(line)
            
            total_data_lines += len(page_data_lines)
            print(f'Page {page_num + 1}: {len(page_data_lines)} data lines')
        
        print(f'\nTotal estimated data lines across first 3 pages: {total_data_lines}')

if __name__ == "__main__":
    examine_pdf()