"""
Model training module for BürgerBot.
Handles translation, summarization, sentiment analysis, and topic modeling.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import joblib
from datetime import datetime
import torch
from transformers import (
    MarianMTModel, MarianTokenizer, 
    BartForConditionalGeneration, BartTokenizer,
    pipeline, AutoTokenizer, AutoModelForSequenceClassification
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from bertopic import BERTopic
import yake
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .config import CONFIG, MODELS_DIR, PROCESSED_DATA_DIR, VISUALIZATIONS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationModel:
    """Handles German to English translation using MarianMT."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["translation"]
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Get the best available device for model inference."""
        if self.config["device"] == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config["device"])
    
    def load_model(self):
        """Load the MarianMT model and tokenizer."""
        try:
            logger.info(f"Loading translation model: {self.config['model_name']}")
            self.tokenizer = MarianTokenizer.from_pretrained(self.config["model_name"])
            self.model = MarianMTModel.from_pretrained(self.config["model_name"])
            self.model.to(self.device)
            self.model.eval()
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation model: {str(e)}")
            raise
    
    def translate_text(self, text: str) -> str:
        """
        Translate German text to English.
        
        Args:
            text: German text to translate
            
        Returns:
            Translated English text
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.config["max_length"], 
                                  truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=self.config["max_length"])
            
            # Decode
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return text  # Return original text if translation fails
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of German texts to English.
        
        Args:
            texts: List of German texts to translate
            
        Returns:
            List of translated English texts
        """
        if self.model is None:
            self.load_model()
        
        translations = []
        batch_size = self.config["batch_size"]
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]
            try:
                # Tokenize batch
                inputs = self.tokenizer(batch, return_tensors="pt", max_length=self.config["max_length"], 
                                      truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate translations
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=self.config["max_length"])
                
                # Decode batch
                batch_translations = [self.tokenizer.decode(output, skip_special_tokens=True) 
                                    for output in outputs]
                translations.extend(batch_translations)
                
            except Exception as e:
                logger.error(f"Error translating batch: {str(e)}")
                translations.extend(batch)  # Return original texts if translation fails
        
        return translations
    
    def save_model(self, model_path: Path = None):
        """Save the trained model."""
        model_path = model_path or MODELS_DIR / "translation_model"
        model_path.mkdir(parents=True, exist_ok=True)
        
        if self.model and self.tokenizer:
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            logger.info(f"Translation model saved to {model_path}")


class SummarizationModel:
    """Handles text summarization using BART."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["summarization"]
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the BART model and tokenizer."""
        try:
            logger.info(f"Loading summarization model: {self.config['model_name']}")
            self.tokenizer = BartTokenizer.from_pretrained(self.config["model_name"])
            self.model = BartForConditionalGeneration.from_pretrained(self.config["model_name"])
            self.model.to(self.device)
            self.model.eval()
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            raise
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize text using BART.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summarized text
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, 
                                  truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config["max_length"],
                    min_length=self.config["min_length"],
                    do_sample=self.config["do_sample"],
                    num_beams=self.config["num_beams"],
                    length_penalty=self.config["length_penalty"],
                    early_stopping=self.config["early_stopping"]
                )
            
            # Decode
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return text[:200] + "..."  # Return truncated text if summarization fails
    
    def summarize_batch(self, texts: List[str]) -> List[str]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: List of texts to summarize
            
        Returns:
            List of summarized texts
        """
        if self.model is None:
            self.load_model()
        
        summaries = []
        
        for text in tqdm(texts, desc="Summarizing"):
            summary = self.summarize_text(text)
            summaries.append(summary)
        
        return summaries


class SentimentAnalyzer:
    """Handles sentiment analysis using pre-trained models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["sentiment"]
        self.pipeline = None
        
    def load_model(self):
        """Load the sentiment analysis pipeline."""
        try:
            logger.info(f"Loading sentiment model: {self.config['model_name']}")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.config["model_name"],
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label and score
        """
        if self.pipeline is None:
            self.load_model()
        
        try:
            result = self.pipeline(text[:self.config["max_length"]])[0]
            return {
                "label": result["label"],
                "score": result["score"],
                "text": text
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5, "text": text}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        if self.pipeline is None:
            self.load_model()
        
        results = []
        batch_size = self.config["batch_size"]
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i + batch_size]
            try:
                batch_results = self.pipeline(batch)
                for j, result in enumerate(batch_results):
                    results.append({
                        "label": result["label"],
                        "score": result["score"],
                        "text": batch[j]
                    })
            except Exception as e:
                logger.error(f"Error analyzing sentiment batch: {str(e)}")
                for text in batch:
                    results.append({"label": "NEUTRAL", "score": 0.5, "text": text})
        
        return results


class TopicModeler:
    """Handles topic modeling using LDA and BERTopic."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["topic_modeling"]
        self.lda_model = None
        self.bertopic_model = None
        self.vectorizer = None
        
    def fit_lda(self, texts: List[str], n_topics: int = None) -> Dict[str, Any]:
        """
        Fit LDA topic model.
        
        Args:
            texts: List of texts to model
            n_topics: Number of topics
            
        Returns:
            Dictionary with model and results
        """
        n_topics = n_topics or self.config["lda"]["n_topics"]
        
        try:
            # Vectorize texts
            self.vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            doc_term_matrix = self.vectorizer.fit_transform(texts)
            
            # Fit LDA
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=self.config["lda"]["max_iter"],
                random_state=self.config["lda"]["random_state"],
                learning_method=self.config["lda"]["learning_method"]
            )
            
            self.lda_model.fit(doc_term_matrix)
            
            # Get topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": topic[top_words_idx].tolist()
                })
            
            return {
                "model": self.lda_model,
                "vectorizer": self.vectorizer,
                "topics": topics,
                "doc_term_matrix": doc_term_matrix
            }
            
        except Exception as e:
            logger.error(f"Error fitting LDA model: {str(e)}")
            return None
    
    def fit_bertopic(self, texts: List[str], n_topics: int = None) -> Dict[str, Any]:
        """
        Fit BERTopic model.
        
        Args:
            texts: List of texts to model
            n_topics: Number of topics
            
        Returns:
            Dictionary with model and results
        """
        n_topics = n_topics or self.config["bertopic"]["n_topics"]
        
        try:
            self.bertopic_model = BERTopic(
                n_topics=n_topics,
                min_topic_size=self.config["bertopic"]["min_topic_size"],
                random_state=self.config["bertopic"]["random_state"]
            )
            
            topics, probs = self.bertopic_model.fit_transform(texts)
            
            # Get topic info
            topic_info = self.bertopic_model.get_topic_info()
            
            return {
                "model": self.bertopic_model,
                "topics": topics,
                "probabilities": probs,
                "topic_info": topic_info
            }
            
        except Exception as e:
            logger.error(f"Error fitting BERTopic model: {str(e)}")
            return None
    
    def get_document_topics(self, texts: List[str], model_type: str = "lda") -> List[Dict[str, Any]]:
        """
        Get topic assignments for documents.
        
        Args:
            texts: List of texts
            model_type: "lda" or "bertopic"
            
        Returns:
            List of topic assignments
        """
        if model_type == "lda" and self.lda_model is not None:
            doc_term_matrix = self.vectorizer.transform(texts)
            topic_distributions = self.lda_model.transform(doc_term_matrix)
            
            results = []
            for i, doc_topics in enumerate(topic_distributions):
                top_topic = doc_topics.argmax()
                results.append({
                    "text": texts[i],
                    "topic_id": int(top_topic),
                    "topic_probability": float(doc_topics[top_topic]),
                    "all_probabilities": doc_topics.tolist()
                })
            
            return results
        
        elif model_type == "bertopic" and self.bertopic_model is not None:
            topics, probs = self.bertopic_model.transform(texts)
            
            results = []
            for i, (topic, prob) in enumerate(zip(topics, probs)):
                results.append({
                    "text": texts[i],
                    "topic_id": int(topic),
                    "topic_probability": float(prob.max()),
                    "all_probabilities": prob.tolist()
                })
            
            return results
        
        else:
            logger.error(f"Model {model_type} not fitted")
            return []


class KeywordExtractor:
    """Handles keyword extraction using YAKE."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["keyword"]
        
    def extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords from text using YAKE.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords with scores
        """
        try:
            # Initialize YAKE
            kw_extractor = yake.KeywordExtractor(
                lan="de",  # German language
                n=1,  # unigrams
                dedupLim=0.9,
                top=self.config["max_keywords"],
                features=None
            )
            
            # Extract keywords
            keywords = kw_extractor.extract_keywords(text)
            
            # Filter and format results
            results = []
            for score, keyword in keywords:
                if len(keyword) >= self.config["min_keyword_length"]:
                    results.append({
                        "keyword": keyword,
                        "score": score,
                        "length": len(keyword)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_keywords_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract keywords from a batch of texts.
        
        Args:
            texts: List of texts to extract keywords from
            
        Returns:
            List of keyword lists for each text
        """
        results = []
        
        for text in tqdm(texts, desc="Extracting keywords"):
            keywords = self.extract_keywords(text)
            results.append(keywords)
        
        return results


class ModelTrainer:
    """Main model training orchestrator."""
    
    def __init__(self):
        self.translation_model = TranslationModel()
        self.summarization_model = SummarizationModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        self.keyword_extractor = KeywordExtractor()
        
    def load_processed_data(self) -> pd.DataFrame:
        """Load processed document data."""
        try:
            # Try to load joblib file first (faster)
            joblib_file = PROCESSED_DATA_DIR / "document_chunks.joblib"
            if joblib_file.exists():
                return joblib.load(joblib_file)
            
            # Fall back to CSV
            csv_file = PROCESSED_DATA_DIR / "document_chunks.csv"
            if csv_file.exists():
                return pd.read_csv(csv_file)
            
            raise FileNotFoundError("No processed data found. Run data preprocessing first.")
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models on the processed data.
        
        Args:
            df: DataFrame with processed documents
            
        Returns:
            Dictionary with all model results
        """
        logger.info("Starting model training pipeline...")
        
        results = {
            "translation": {},
            "summarization": {},
            "sentiment": {},
            "topic_modeling": {},
            "keywords": {},
            "metadata": {
                "total_documents": len(df),
                "training_date": datetime.now().isoformat()
            }
        }
        
        # Get German texts for processing
        german_texts = df["chunk_text"].tolist()
        
        # Step 1: Translation
        logger.info("Training translation model...")
        try:
            translated_texts = self.translation_model.translate_batch(german_texts)
            results["translation"] = {
                "translated_texts": translated_texts,
                "model_info": {
                    "model_name": self.translation_model.config["model_name"],
                    "device": str(self.translation_model.device)
                }
            }
            logger.info("Translation completed")
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            translated_texts = german_texts  # Use original texts as fallback
        
        # Step 2: Summarization (on translated texts)
        logger.info("Training summarization model...")
        try:
            summaries = self.summarization_model.summarize_batch(translated_texts)
            results["summarization"] = {
                "summaries": summaries,
                "model_info": {
                    "model_name": self.summarization_model.config["model_name"]
                }
            }
            logger.info("Summarization completed")
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
        
        # Step 3: Sentiment Analysis
        logger.info("Training sentiment analysis model...")
        try:
            sentiment_results = self.sentiment_analyzer.analyze_batch(translated_texts)
            results["sentiment"] = {
                "sentiment_results": sentiment_results,
                "model_info": {
                    "model_name": self.sentiment_analyzer.config["model_name"]
                }
            }
            logger.info("Sentiment analysis completed")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
        
        # Step 4: Topic Modeling
        logger.info("Training topic models...")
        try:
            # LDA
            lda_results = self.topic_modeler.fit_lda(translated_texts)
            if lda_results:
                results["topic_modeling"]["lda"] = lda_results
            
            # BERTopic
            bertopic_results = self.topic_modeler.fit_bertopic(translated_texts)
            if bertopic_results:
                results["topic_modeling"]["bertopic"] = bertopic_results
            
            logger.info("Topic modeling completed")
        except Exception as e:
            logger.error(f"Topic modeling failed: {str(e)}")
        
        # Step 5: Keyword Extraction
        logger.info("Extracting keywords...")
        try:
            keywords = self.keyword_extractor.extract_keywords_batch(german_texts)
            results["keywords"] = {
                "keywords": keywords,
                "model_info": {
                    "model": "YAKE",
                    "language": "German"
                }
            }
            logger.info("Keyword extraction completed")
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
        
        # Save results
        self.save_results(results)
        
        logger.info("Model training pipeline completed!")
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save model results and trained models."""
        # Save results
        results_file = MODELS_DIR / "training_results.joblib"
        joblib.dump(results, results_file)
        logger.info(f"Training results saved to {results_file}")
        
        # Save individual models
        if results.get("translation"):
            self.translation_model.save_model()
        
        # Save topic models
        if results.get("topic_modeling"):
            if "lda" in results["topic_modeling"]:
                lda_file = MODELS_DIR / "lda_model.joblib"
                joblib.dump(results["topic_modeling"]["lda"], lda_file)
            
            if "bertopic" in results["topic_modeling"]:
                bertopic_file = MODELS_DIR / "bertopic_model.joblib"
                joblib.dump(results["topic_modeling"]["bertopic"], lda_file)


def main():
    """Main training pipeline."""
    trainer = ModelTrainer()
    
    # Load data
    logger.info("Loading processed data...")
    df = trainer.load_processed_data()
    
    # Train models
    results = trainer.train_all_models(df)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 