"""
Unit tests for model training module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.model_training import TranslationModel, SummarizationModel, SentimentAnalyzer, TopicModeler, KeywordExtractor
from src.config import CONFIG


class TestTranslationModel(unittest.TestCase):
    """Test cases for TranslationModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TranslationModel()
    
    def test_device_selection(self):
        """Test device selection logic."""
        # Test auto device selection
        device = self.model._get_device()
        self.assertIn(device.type, ['cpu', 'cuda'])
        
        # Test manual device selection
        self.model.config["device"] = "cpu"
        device = self.model._get_device()
        self.assertEqual(device.type, 'cpu')
    
    @patch('src.model_training.MarianTokenizer.from_pretrained')
    @patch('src.model_training.MarianMTModel.from_pretrained')
    def test_load_model(self, mock_model, mock_tokenizer):
        """Test model loading."""
        # Mock the model and tokenizer
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        
        self.model.load_model()
        
        # Check that the model and tokenizer were called
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
    
    def test_translate_text_error_handling(self):
        """Test translation error handling."""
        # Test with None model (should return original text)
        result = self.model.translate_text("Test text")
        self.assertEqual(result, "Test text")


class TestSummarizationModel(unittest.TestCase):
    """Test cases for SummarizationModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SummarizationModel()
    
    @patch('src.model_training.BartTokenizer.from_pretrained')
    @patch('src.model_training.BartForConditionalGeneration.from_pretrained')
    def test_load_model(self, mock_model, mock_tokenizer):
        """Test model loading."""
        # Mock the model and tokenizer
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        
        self.model.load_model()
        
        # Check that the model and tokenizer were called
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
    
    def test_summarize_text_error_handling(self):
        """Test summarization error handling."""
        # Test with None model (should return truncated text)
        result = self.model.summarize_text("This is a test text for summarization.")
        self.assertIn("This is a test text for summarization", result)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    @patch('src.model_training.pipeline')
    def test_load_model(self, mock_pipeline):
        """Test model loading."""
        # Mock the pipeline
        mock_pipeline.return_value = Mock()
        
        self.analyzer.load_model()
        
        # Check that the pipeline was called
        mock_pipeline.assert_called_once()
    
    def test_analyze_sentiment_error_handling(self):
        """Test sentiment analysis error handling."""
        # Test with None pipeline (should return neutral sentiment)
        result = self.analyzer.analyze_sentiment("Test text")
        self.assertEqual(result["label"], "NEUTRAL")
        self.assertEqual(result["score"], 0.5)


class TestTopicModeler(unittest.TestCase):
    """Test cases for TopicModeler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.modeler = TopicModeler()
    
    def test_fit_lda(self):
        """Test LDA topic modeling."""
        # Create sample texts
        texts = [
            "This is a document about politics and government.",
            "Another document about economics and finance.",
            "A third document about technology and innovation.",
            "Politics and economics are important topics.",
            "Technology drives innovation in government."
        ]
        
        # Test LDA fitting
        result = self.modeler.fit_lda(texts, n_topics=3)
        
        if result is not None:
            # Check that the result has the expected structure
            self.assertIn("model", result)
            self.assertIn("vectorizer", result)
            self.assertIn("topics", result)
            self.assertIn("doc_term_matrix", result)
            
            # Check topics
            self.assertEqual(len(result["topics"]), 3)
            
            # Check topic structure
            for topic in result["topics"]:
                self.assertIn("topic_id", topic)
                self.assertIn("words", topic)
                self.assertIn("weights", topic)
    
    @patch('src.model_training.BERTopic')
    def test_fit_bertopic(self, mock_bertopic):
        """Test BERTopic modeling."""
        # Mock BERTopic
        mock_model = Mock()
        mock_model.fit_transform.return_value = (np.array([0, 1, 2]), np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))
        mock_model.get_topic_info.return_value = pd.DataFrame({
            "Topic": [0, 1, 2],
            "Count": [10, 8, 6],
            "Name": ["Topic 0", "Topic 1", "Topic 2"]
        })
        mock_bertopic.return_value = mock_model
        
        # Create sample texts
        texts = [
            "This is a document about politics and government.",
            "Another document about economics and finance.",
            "A third document about technology and innovation."
        ]
        
        # Test BERTopic fitting
        result = self.modeler.fit_bertopic(texts, n_topics=3)
        
        if result is not None:
            # Check that the result has the expected structure
            self.assertIn("model", result)
            self.assertIn("topics", result)
            self.assertIn("probabilities", result)
            self.assertIn("topic_info", result)


class TestKeywordExtractor(unittest.TestCase):
    """Test cases for KeywordExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = KeywordExtractor()
    
    @patch('src.model_training.yake.KeywordExtractor')
    def test_extract_keywords(self, mock_yake):
        """Test keyword extraction."""
        # Mock YAKE
        mock_extractor = Mock()
        mock_extractor.extract_keywords.return_value = [
            (0.1, "politics"),
            (0.2, "government"),
            (0.3, "economics")
        ]
        mock_yake.return_value = mock_extractor
        
        # Test keyword extraction
        text = "This is a document about politics and government economics."
        keywords = self.extractor.extract_keywords(text)
        
        # Check that keywords were extracted
        self.assertIsInstance(keywords, list)
        
        if keywords:
            # Check keyword structure
            for keyword in keywords:
                self.assertIn("keyword", keyword)
                self.assertIn("score", keyword)
                self.assertIn("length", keyword)
    
    def test_extract_keywords_error_handling(self):
        """Test keyword extraction error handling."""
        # Test with empty text
        keywords = self.extractor.extract_keywords("")
        self.assertEqual(keywords, [])
        
        # Test with very short text
        keywords = self.extractor.extract_keywords("Hi")
        self.assertEqual(keywords, [])


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_processed_data(self):
        """Test loading processed data."""
        # Create sample data
        sample_data = {
            "filename": ["test1.pdf", "test2.pdf"],
            "page": [1, 1],
            "chunk_text": ["Text 1", "Text 2"],
            "chunk_length": [100, 150]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Save to temp directory
        temp_file = Path(self.temp_dir) / "document_chunks.joblib"
        import joblib
        joblib.dump(df, temp_file)
        
        # Test loading (with mocked path)
        with patch('src.model_training.PROCESSED_DATA_DIR', Path(self.temp_dir)):
            loaded_df = self.trainer.load_processed_data()
            self.assertEqual(len(loaded_df), len(df))
            self.assertTrue(all(loaded_df.columns == df.columns))


if __name__ == "__main__":
    unittest.main() 