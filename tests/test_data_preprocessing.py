"""
Unit tests for data preprocessing module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_preprocessing import PDFProcessor, DataPreprocessor
from src.config import CONFIG


class TestPDFProcessor(unittest.TestCase):
    """Test cases for PDFProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = PDFProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test URL removal
        text_with_url = "This is a test with http://example.com URL"
        cleaned = self.processor.clean_text(text_with_url)
        self.assertNotIn("http://example.com", cleaned)
        
        # Test email removal
        text_with_email = "Contact us at test@example.com"
        cleaned = self.processor.clean_text(text_with_email)
        self.assertNotIn("test@example.com", cleaned)
        
        # Test whitespace normalization
        text_with_whitespace = "  Multiple    spaces   "
        cleaned = self.processor.clean_text(text_with_whitespace)
        self.assertEqual(cleaned, "multiple spaces")
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        # Test short text
        short_text = "This is a short text."
        chunks = self.processor.chunk_text(short_text, chunk_size=100, overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)
        
        # Test long text
        long_text = "This is a longer text. " * 50  # ~1000 characters
        chunks = self.processor.chunk_text(long_text, chunk_size=200, overlap=50)
        self.assertGreater(len(chunks), 1)
        
        # Check overlap
        for i in range(1, len(chunks)):
            # There should be some overlap between consecutive chunks
            self.assertTrue(len(set(chunks[i-1].split()) & set(chunks[i].split())) > 0)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_document_chunks(self):
        """Test document chunking functionality."""
        # Create sample data
        sample_data = {
            "filename": ["test1.pdf", "test2.pdf"],
            "page": [1, 1],
            "cleaned_text": [
                "This is a test document with some content. " * 20,  # ~600 chars
                "Another test document with different content. " * 15  # ~450 chars
            ],
            "chunk_length": [600, 450],
            "extraction_date": ["2023-01-01", "2023-01-01"]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test chunking
        chunked_df = self.preprocessor.create_document_chunks(df, chunk_size=200, overlap=50)
        
        # Check that chunks were created
        self.assertGreater(len(chunked_df), len(df))
        
        # Check required columns
        required_columns = ["filename", "page", "chunk_index", "chunk_text", "chunk_length", "extraction_date"]
        for col in required_columns:
            self.assertIn(col, chunked_df.columns)
        
        # Check chunk indices
        self.assertTrue(all(chunked_df["chunk_index"] >= 0))
    
    def test_filter_documents_by_length(self):
        """Test document filtering functionality."""
        # Create sample data with varying lengths
        sample_data = {
            "chunk_length": [50, 100, 500, 1000, 5000, 10000]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test filtering
        filtered_df = self.preprocessor.pdf_processor.chunk_text = lambda x, **kwargs: [x]
        
        # This is a placeholder test - in a real implementation,
        # you would test the actual filtering logic
        self.assertTrue(len(filtered_df) <= len(df))


class TestWebScraper(unittest.TestCase):
    """Test cases for WebScraper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.data_preprocessing import WebScraper
        self.scraper = WebScraper()
    
    def test_extract_pdf_links(self):
        """Test PDF link extraction from HTML."""
        from bs4 import BeautifulSoup
        
        # Create sample HTML with PDF links
        html_content = """
        <html>
            <body>
                <a href="document1.pdf">PDF Document 1</a>
                <a href="https://example.com/document2.pdf">PDF Document 2</a>
                <a href="document3.txt">Text Document</a>
                <a href="document4.pdf">PDF Document 4</a>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html_content, 'html.parser')
        base_url = "https://example.com"
        
        pdf_links = self.scraper._extract_pdf_links(soup, base_url)
        
        # Should find 3 PDF links
        self.assertEqual(len(pdf_links), 3)
        
        # Check that all links are PDFs
        for link in pdf_links:
            self.assertTrue(link["filename"].lower().endswith('.pdf'))
            self.assertIn("url", link)
            self.assertIn("title", link)


if __name__ == "__main__":
    unittest.main() 