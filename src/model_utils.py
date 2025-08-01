"""
Model utilities for BürgerBot.
Handles model loading, saving, and inference utilities.
"""

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import MarianMTModel, MarianTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from .config import CONFIG, MODELS_DIR, VISUALIZATIONS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading trained models."""
    
    @staticmethod
    def load_training_results() -> Dict[str, Any]:
        """Load training results from disk."""
        results_file = MODELS_DIR / "training_results.joblib"
        if results_file.exists():
            return joblib.load(results_file)
        else:
            raise FileNotFoundError("Training results not found. Run model training first.")
    
    @staticmethod
    def load_translation_model() -> tuple:
        """Load translation model and tokenizer."""
        model_path = MODELS_DIR / "translation_model"
        if model_path.exists():
            tokenizer = MarianTokenizer.from_pretrained(str(model_path))
            model = MarianMTModel.from_pretrained(str(model_path))
            return model, tokenizer
        else:
            raise FileNotFoundError("Translation model not found. Run model training first.")
    
    @staticmethod
    def load_lda_model() -> Dict[str, Any]:
        """Load LDA topic model."""
        lda_file = MODELS_DIR / "lda_model.joblib"
        if lda_file.exists():
            return joblib.load(lda_file)
        else:
            raise FileNotFoundError("LDA model not found. Run model training first.")
    
    @staticmethod
    def load_bertopic_model() -> Dict[str, Any]:
        """Load BERTopic model."""
        bertopic_file = MODELS_DIR / "bertopic_model.joblib"
        if bertopic_file.exists():
            return joblib.load(bertopic_file)
        else:
            raise FileNotFoundError("BERTopic model not found. Run model training first.")


class VisualizationUtils:
    """Utility class for creating visualizations."""
    
    @staticmethod
    def create_wordcloud(text_data: Union[str, List[str]], title: str = "Word Cloud", 
                        save_path: Path = None) -> WordCloud:
        """
        Create a word cloud from text data.
        
        Args:
            text_data: Text or list of texts
            title: Title for the word cloud
            save_path: Path to save the visualization
            
        Returns:
            WordCloud object
        """
        config = CONFIG["visualization"]["wordcloud"]
        
        # Combine text if list
        if isinstance(text_data, list):
            text = " ".join(text_data)
        else:
            text = text_data
        
        # Create word cloud
        wordcloud = WordCloud(
            width=config["width"],
            height=config["height"],
            background_color=config["background_color"],
            max_words=config["max_words"],
            relative_scaling=config["relative_scaling"]
        ).generate(text)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = save_path or VISUALIZATIONS_DIR / f"{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Word cloud saved to {save_path}")
        
        return wordcloud
    
    @staticmethod
    def plot_sentiment_distribution(sentiment_results: List[Dict[str, Any]], 
                                  save_path: Path = None) -> go.Figure:
        """
        Create sentiment distribution plot.
        
        Args:
            sentiment_results: List of sentiment analysis results
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure
        """
        # Extract sentiment labels and scores
        labels = [result["label"] for result in sentiment_results]
        scores = [result["score"] for result in sentiment_results]
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Sentiment Distribution", "Sentiment Scores"),
            specs=[[{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Pie chart
        label_counts = pd.Series(labels).value_counts()
        fig.add_trace(
            go.Pie(labels=label_counts.index, values=label_counts.values, name="Distribution"),
            row=1, col=1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=scores, nbinsx=20, name="Scores"),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Sentiment Analysis Results",
            showlegend=False,
            height=500
        )
        
        # Save if path provided
        if save_path:
            save_path = save_path or VISUALIZATIONS_DIR / "sentiment_analysis.png"
            fig.write_image(str(save_path))
            logger.info(f"Sentiment plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_topic_distribution(topic_results: Dict[str, Any], model_type: str = "lda",
                              save_path: Path = None) -> go.Figure:
        """
        Create topic distribution plot.
        
        Args:
            topic_results: Topic modeling results
            model_type: "lda" or "bertopic"
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure
        """
        if model_type == "lda":
            topics = topic_results["topics"]
            
            # Create bar chart of topic distributions
            topic_ids = [topic["topic_id"] for topic in topics]
            topic_sizes = [len(topic["words"]) for topic in topics]
            
            fig = go.Figure(data=[
                go.Bar(x=topic_ids, y=topic_sizes, text=topic_sizes, textposition='auto')
            ])
            
            fig.update_layout(
                title="LDA Topic Distribution",
                xaxis_title="Topic ID",
                yaxis_title="Number of Words",
                height=500
            )
            
        elif model_type == "bertopic":
            topic_info = topic_results["topic_info"]
            
            # Create bar chart of topic counts
            fig = go.Figure(data=[
                go.Bar(x=topic_info["Topic"], y=topic_info["Count"], 
                      text=topic_info["Count"], textposition='auto')
            ])
            
            fig.update_layout(
                title="BERTopic Distribution",
                xaxis_title="Topic ID",
                yaxis_title="Document Count",
                height=500
            )
        
        # Save if path provided
        if save_path:
            save_path = save_path or VISUALIZATIONS_DIR / f"{model_type}_topic_distribution.png"
            fig.write_image(str(save_path))
            logger.info(f"Topic plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_keyword_trends(keyword_results: List[List[Dict[str, Any]]], 
                           save_path: Path = None) -> go.Figure:
        """
        Create keyword trends plot.
        
        Args:
            keyword_results: List of keyword extraction results
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure
        """
        # Flatten and aggregate keywords
        all_keywords = []
        for doc_keywords in keyword_results:
            all_keywords.extend(doc_keywords)
        
        # Create DataFrame
        df = pd.DataFrame(all_keywords)
        
        if len(df) == 0:
            logger.warning("No keywords found for plotting")
            return go.Figure()
        
        # Get top keywords by frequency
        top_keywords = df["keyword"].value_counts().head(20)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(x=top_keywords.values, y=top_keywords.index, orientation='h')
        ])
        
        fig.update_layout(
            title="Top Keywords by Frequency",
            xaxis_title="Frequency",
            yaxis_title="Keyword",
            height=600
        )
        
        # Save if path provided
        if save_path:
            save_path = save_path or VISUALIZATIONS_DIR / "keyword_trends.png"
            fig.write_image(str(save_path))
            logger.info(f"Keyword plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_document_lengths(df: pd.DataFrame, save_path: Path = None) -> go.Figure:
        """
        Create document length distribution plot.
        
        Args:
            df: DataFrame with document data
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Document Length Distribution", "Document Length by File"),
            specs=[[{"type": "histogram"}, {"type": "box"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df["chunk_length"], nbinsx=30, name="Length Distribution"),
            row=1, col=1
        )
        
        # Box plot by file
        fig.add_trace(
            go.Box(x=df["filename"], y=df["chunk_length"], name="Length by File"),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Document Length Analysis",
            showlegend=False,
            height=500
        )
        
        # Save if path provided
        if save_path:
            save_path = save_path or VISUALIZATIONS_DIR / "document_lengths.png"
            fig.write_image(str(save_path))
            logger.info(f"Document length plot saved to {save_path}")
        
        return fig


class InferenceUtils:
    """Utility class for model inference."""
    
    @staticmethod
    def translate_text(text: str, model: MarianMTModel = None, 
                      tokenizer: MarianTokenizer = None) -> str:
        """
        Translate German text to English.
        
        Args:
            text: German text to translate
            model: Pre-loaded translation model
            tokenizer: Pre-loaded tokenizer
            
        Returns:
            Translated English text
        """
        try:
            if model is None or tokenizer is None:
                model, tokenizer = ModelLoader.load_translation_model()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", max_length=512, 
                             truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512)
            
            # Decode
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
    
    @staticmethod
    def get_topic_for_text(text: str, model_type: str = "lda") -> Dict[str, Any]:
        """
        Get topic assignment for a text.
        
        Args:
            text: Text to analyze
            model_type: "lda" or "bertopic"
            
        Returns:
            Topic assignment dictionary
        """
        try:
            if model_type == "lda":
                lda_results = ModelLoader.load_lda_model()
                vectorizer = lda_results["vectorizer"]
                model = lda_results["model"]
                
                # Vectorize text
                doc_term_matrix = vectorizer.transform([text])
                topic_distribution = model.transform(doc_term_matrix)[0]
                
                top_topic = topic_distribution.argmax()
                topics = lda_results["topics"]
                
                return {
                    "topic_id": int(top_topic),
                    "topic_probability": float(topic_distribution[top_topic]),
                    "topic_words": topics[top_topic]["words"],
                    "all_probabilities": topic_distribution.tolist()
                }
            
            elif model_type == "bertopic":
                bertopic_results = ModelLoader.load_bertopic_model()
                model = bertopic_results["model"]
                
                # Get topic
                topics, probs = model.transform([text])
                topic = topics[0]
                prob = probs[0]
                
                return {
                    "topic_id": int(topic),
                    "topic_probability": float(prob.max()),
                    "all_probabilities": prob.tolist()
                }
            
        except Exception as e:
            logger.error(f"Topic analysis error: {str(e)}")
            return {"topic_id": -1, "topic_probability": 0.0, "error": str(e)}
    
    @staticmethod
    def extract_keywords_from_text(text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords with scores
        """
        try:
            import yake
            
            config = CONFIG["keyword"]
            kw_extractor = yake.KeywordExtractor(
                lan="de",
                n=1,
                dedupLim=0.9,
                top=config["max_keywords"],
                features=None
            )
            
            keywords = kw_extractor.extract_keywords(text)
            
            results = []
            for score, keyword in keywords:
                if len(keyword) >= config["min_keyword_length"]:
                    results.append({
                        "keyword": keyword,
                        "score": score,
                        "length": len(keyword)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            return []


class DataUtils:
    """Utility class for data manipulation."""
    
    @staticmethod
    def load_processed_data() -> pd.DataFrame:
        """Load processed document data."""
        try:
            # Try joblib first
            joblib_file = CONFIG["PROCESSED_DATA_DIR"] / "document_chunks.joblib"
            if joblib_file.exists():
                return joblib.load(joblib_file)
            
            # Fall back to CSV
            csv_file = CONFIG["PROCESSED_DATA_DIR"] / "document_chunks.csv"
            if csv_file.exists():
                return pd.read_csv(csv_file)
            
            raise FileNotFoundError("No processed data found")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    @staticmethod
    def filter_documents_by_length(df: pd.DataFrame, min_length: int = 100, 
                                 max_length: int = 5000) -> pd.DataFrame:
        """
        Filter documents by text length.
        
        Args:
            df: DataFrame with documents
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered DataFrame
        """
        return df[
            (df["chunk_length"] >= min_length) & 
            (df["chunk_length"] <= max_length)
        ]
    
    @staticmethod
    def sample_documents(df: pd.DataFrame, n_samples: int = 100, 
                        random_state: int = 42) -> pd.DataFrame:
        """
        Sample documents for analysis.
        
        Args:
            df: DataFrame with documents
            n_samples: Number of samples to take
            random_state: Random seed
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= n_samples:
            return df
        
        return df.sample(n=n_samples, random_state=random_state)
    
    @staticmethod
    def get_document_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic statistics about documents.
        
        Args:
            df: DataFrame with documents
            
        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(df),
            "unique_files": df["filename"].nunique(),
            "total_pages": df["page"].sum(),
            "avg_chunk_length": df["chunk_length"].mean(),
            "min_chunk_length": df["chunk_length"].min(),
            "max_chunk_length": df["chunk_length"].max(),
            "std_chunk_length": df["chunk_length"].std(),
            "total_characters": df["chunk_length"].sum()
        }


def main():
    """Main utility functions demo."""
    try:
        # Load data
        df = DataUtils.load_processed_data()
        stats = DataUtils.get_document_statistics(df)
        logger.info(f"Document statistics: {stats}")
        
        # Create visualizations
        VisualizationUtils.plot_document_lengths(df)
        
        # Sample documents for analysis
        sample_df = DataUtils.sample_documents(df, n_samples=50)
        
        # Demo inference
        if len(sample_df) > 0:
            sample_text = sample_df.iloc[0]["chunk_text"]
            translation = InferenceUtils.translate_text(sample_text)
            keywords = InferenceUtils.extract_keywords_from_text(sample_text)
            
            logger.info(f"Sample translation: {translation[:100]}...")
            logger.info(f"Sample keywords: {keywords[:5]}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main() 