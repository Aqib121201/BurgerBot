"""
Data preprocessing module for BürgerBot.
Handles PDF extraction, text cleaning, and data preparation.
"""

import logging
import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from datetime import datetime

from .config import CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF document processing and text extraction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["pdf"]
        
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and tables
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            tables = []
            metadata = {
                "filename": pdf_path.name,
                "total_pages": len(doc),
                "processed_pages": 0,
                "extraction_date": datetime.now().isoformat()
            }
            
            # Process pages up to max_pages limit
            max_pages = min(self.config["max_pages"], len(doc))
            
            for page_num in range(max_pages):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if len(text.strip()) >= self.config["min_text_length"]:
                    text_content.append({
                        "page": page_num + 1,
                        "text": text.strip(),
                        "bbox": page.rect
                    })
                
                # Extract tables if enabled
                if self.config["extract_tables"]:
                    page_tables = page.get_tables()
                    for table_idx, table in enumerate(page_tables):
                        tables.append({
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "data": table
                        })
                
                metadata["processed_pages"] += 1
            
            doc.close()
            
            return {
                "text_content": text_content,
                "tables": tables,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        config = CONFIG["data"]["text_cleaning"]
        
        # Remove URLs
        if config["remove_urls"]:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emails
        if config["remove_emails"]:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional)
        if config["remove_numbers"]:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation (optional)
        if config["remove_punctuation"]:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        if config["lowercase"]:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or CONFIG["data"]["chunk_size"]
        overlap = overlap or CONFIG["data"]["overlap"]
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks


class WebScraper:
    """Handles web scraping of German government websites."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["scraping"]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config["user_agent"]})
        self.visited_urls = set()
        
    def scrape_pdfs_from_url(self, base_url: str, max_depth: int = None) -> List[Dict[str, Any]]:
        """
        Scrape PDF links from a government website.
        
        Args:
            base_url: Base URL to start scraping
            max_depth: Maximum depth for crawling
            
        Returns:
            List of PDF metadata and download links
        """
        max_depth = max_depth or self.config["max_depth"]
        pdf_links = []
        
        try:
            response = self.session.get(base_url, timeout=self.config["timeout"])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_links.extend(self._extract_pdf_links(soup, base_url))
            
            # Follow links up to max_depth
            if max_depth > 0:
                links = soup.find_all('a', href=True)
                for link in links[:10]:  # Limit to first 10 links to avoid overwhelming
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    
                    if (full_url not in self.visited_urls and 
                        urlparse(full_url).netloc == urlparse(base_url).netloc):
                        self.visited_urls.add(full_url)
                        time.sleep(self.config["delay_between_requests"])
                        
                        try:
                            sub_pdfs = self.scrape_pdfs_from_url(full_url, max_depth - 1)
                            pdf_links.extend(sub_pdfs)
                        except Exception as e:
                            logger.warning(f"Error scraping {full_url}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error scraping {base_url}: {str(e)}")
        
        return pdf_links
    
    def _extract_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract PDF links from a BeautifulSoup object."""
        pdf_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                full_url = urljoin(base_url, href)
                pdf_links.append({
                    "url": full_url,
                    "filename": href.split('/')[-1],
                    "title": link.get_text(strip=True),
                    "source_url": base_url,
                    "scraped_date": datetime.now().isoformat()
                })
        
        return pdf_links
    
    def download_pdf(self, pdf_info: Dict[str, Any], download_dir: Path) -> Optional[Path]:
        """
        Download a PDF file.
        
        Args:
            pdf_info: PDF metadata dictionary
            download_dir: Directory to save the PDF
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        try:
            response = self.session.get(pdf_info["url"], timeout=self.config["timeout"])
            response.raise_for_status()
            
            filename = pdf_info["filename"]
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
            
            file_path = download_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading {pdf_info['url']}: {str(e)}")
            return None


class DataPreprocessor:
    """Main data preprocessing orchestrator."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.web_scraper = WebScraper()
        
    def process_pdf_directory(self, input_dir: Path, output_dir: Path = None) -> pd.DataFrame:
        """
        Process all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed data
            
        Returns:
            DataFrame with processed document data
        """
        output_dir = output_dir or PROCESSED_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_docs = []
        
        for pdf_file in input_dir.glob("*.pdf"):
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract text from PDF
            extraction_result = self.pdf_processor.extract_text_from_pdf(pdf_file)
            
            if extraction_result:
                # Process each page
                for page_data in extraction_result["text_content"]:
                    cleaned_text = self.pdf_processor.clean_text(page_data["text"])
                    
                    if len(cleaned_text) > 0:
                        processed_docs.append({
                            "filename": pdf_file.name,
                            "page": page_data["page"],
                            "raw_text": page_data["text"],
                            "cleaned_text": cleaned_text,
                            "text_length": len(cleaned_text),
                            "extraction_date": extraction_result["metadata"]["extraction_date"],
                            "total_pages": extraction_result["metadata"]["total_pages"]
                        })
        
        # Create DataFrame
        df = pd.DataFrame(processed_docs)
        
        # Save processed data
        output_file = output_dir / "processed_documents.csv"
        df.to_csv(output_file, index=False)
        
        # Save as joblib for faster loading
        joblib_file = output_dir / "processed_documents.joblib"
        joblib.dump(df, joblib_file)
        
        logger.info(f"Processed {len(df)} document pages. Saved to {output_file}")
        return df
    
    def scrape_and_download_pdfs(self, output_dir: Path = None) -> List[Path]:
        """
        Scrape and download PDFs from German government websites.
        
        Args:
            output_dir: Directory to save downloaded PDFs
            
        Returns:
            List of downloaded PDF paths
        """
        output_dir = output_dir or RAW_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        for base_url in self.web_scraper.config["base_urls"]:
            logger.info(f"Scraping PDFs from: {base_url}")
            
            pdf_links = self.web_scraper.scrape_pdfs_from_url(base_url)
            
            # Download PDFs in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_pdf = {
                    executor.submit(self.web_scraper.download_pdf, pdf_info, output_dir): pdf_info
                    for pdf_info in pdf_links
                }
                
                for future in as_completed(future_to_pdf):
                    pdf_info = future_to_pdf[future]
                    try:
                        downloaded_path = future.result()
                        if downloaded_path:
                            downloaded_files.append(downloaded_path)
                    except Exception as e:
                        logger.error(f"Error downloading {pdf_info['url']}: {str(e)}")
        
        logger.info(f"Downloaded {len(downloaded_files)} PDF files")
        return downloaded_files
    
    def create_document_chunks(self, df: pd.DataFrame, chunk_size: int = None, overlap: int = None) -> pd.DataFrame:
        """
        Create overlapping chunks from document text.
        
        Args:
            df: DataFrame with document data
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            DataFrame with chunked documents
        """
        chunked_docs = []
        
        for _, row in df.iterrows():
            chunks = self.pdf_processor.chunk_text(
                row["cleaned_text"], 
                chunk_size=chunk_size, 
                overlap=overlap
            )
            
            for chunk_idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    "filename": row["filename"],
                    "page": row["page"],
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk,
                    "chunk_length": len(chunk),
                    "extraction_date": row["extraction_date"]
                })
        
        chunked_df = pd.DataFrame(chunked_docs)
        
        # Save chunked data
        output_file = PROCESSED_DATA_DIR / "document_chunks.csv"
        chunked_df.to_csv(output_file, index=False)
        
        joblib_file = PROCESSED_DATA_DIR / "document_chunks.joblib"
        joblib.dump(chunked_df, joblib_file)
        
        logger.info(f"Created {len(chunked_df)} document chunks")
        return chunked_df


def main():
    """Main preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    
    # Step 1: Scrape and download PDFs
    logger.info("Starting PDF scraping and download...")
    downloaded_files = preprocessor.scrape_and_download_pdfs()
    
    # Step 2: Process downloaded PDFs
    if downloaded_files:
        logger.info("Processing downloaded PDFs...")
        df = preprocessor.process_pdf_directory(RAW_DATA_DIR)
        
        # Step 3: Create document chunks
        logger.info("Creating document chunks...")
        chunked_df = preprocessor.create_document_chunks(df)
        
        logger.info("Data preprocessing completed successfully!")
    else:
        logger.warning("No PDFs were downloaded. Check your internet connection and website availability.")


if __name__ == "__main__":
    main() 