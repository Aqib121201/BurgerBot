"""
Configuration settings for BürgerBot NLP pipeline.
Centralized hyperparameters, paths, and model configurations.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                 MODELS_DIR, VISUALIZATIONS_DIR, LOGS_DIR, CONFIGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# PDF Processing Configuration
PDF_CONFIG = {
    "max_pages": 100,  # Maximum pages to process per PDF
    "min_text_length": 50,  # Minimum text length to consider
    "extract_images": False,  # Whether to extract images
    "extract_tables": True,  # Whether to extract tables
}

# Translation Configuration
TRANSLATION_CONFIG = {
    "source_language": "de",  # German
    "target_language": "en",  # English
    "model_name": "Helsinki-NLP/opus-mt-de-en",  # MarianMT model
    "max_length": 512,
    "batch_size": 8,
    "device": "auto",  # "cpu", "cuda", or "auto"
}

# Summarization Configuration
SUMMARIZATION_CONFIG = {
    "model_name": "facebook/bart-large-cnn",  # For English summaries
    "max_length": 150,
    "min_length": 30,
    "do_sample": False,
    "num_beams": 4,
    "length_penalty": 2.0,
    "early_stopping": True,
}

# Sentiment Analysis Configuration
SENTIMENT_CONFIG = {
    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "batch_size": 16,
    "max_length": 512,
}

# Topic Modeling Configuration
TOPIC_MODELING_CONFIG = {
    "lda": {
        "n_topics": 10,
        "max_iter": 100,
        "random_state": 42,
        "learning_method": "online",
    },
    "bertopic": {
        "n_topics": 10,
        "min_topic_size": 10,
        "random_state": 42,
    }
}

# Keyword Extraction Configuration
KEYWORD_CONFIG = {
    "max_keywords": 20,
    "min_keyword_length": 3,
    "stop_words": ["der", "die", "das", "und", "in", "zu", "den", "von", "mit", "sich", "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als", "auch", "es", "an", "werden", "aus", "er", "hat", "daß", "sie", "nach", "wird", "bei", "einer", "um", "so", "zum", "war", "haben", "oder", "aber", "vor", "zur", "bis", "mehr", "durch", "man", "sein", "wurde", "sei", "In", "Prozent", "hatte", "kann", "gegen", "vom", "können", "schon", "wenn", "habe", "seine", "Mark", "ihre", "dann", "unter", "wir", "soll", "ich", "eines", "Es", "Jahr", "zwei", "Jahren", "diese", "dieser", "wieder", "keine", "Uhr", "seiner", "worden", "Und", "will", "zwischen", "Im", "immer", "Millionen", "Ein", "was", "sagte"],
}

# Web Scraping Configuration
SCRAPING_CONFIG = {
    "base_urls": [
        "https://www.bundestag.de",
        "https://www.bundesregierung.de",
        "https://www.bundesrat.de",
    ],
    "max_depth": 3,
    "delay_between_requests": 1.0,  # seconds
    "user_agent": "BürgerBot/1.0 (Research Project)",
    "timeout": 30,
}

# Data Processing Configuration
DATA_CONFIG = {
    "text_cleaning": {
        "remove_urls": True,
        "remove_emails": True,
        "remove_numbers": False,
        "remove_punctuation": False,
        "lowercase": True,
        "remove_stopwords": True,
    },
    "chunk_size": 1000,  # For processing large documents
    "overlap": 200,  # Overlap between chunks
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "wordcloud": {
        "width": 800,
        "height": 600,
        "background_color": "white",
        "max_words": 100,
        "relative_scaling": 0.5,
    },
    "charts": {
        "theme": "plotly_white",
        "height": 500,
        "width": 800,
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "burgerbot.log",
}

# Model Storage Configuration
MODEL_STORAGE = {
    "format": "joblib",  # "joblib" or "pickle"
    "compression": True,
    "versioning": True,
}

# API Configuration (for future deployment)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 4,
}

# Cache Configuration
CACHE_CONFIG = {
    "enable": True,
    "ttl": 3600,  # 1 hour
    "max_size": 1000,
}

# All configurations combined
CONFIG = {
    "pdf": PDF_CONFIG,
    "translation": TRANSLATION_CONFIG,
    "summarization": SUMMARIZATION_CONFIG,
    "sentiment": SENTIMENT_CONFIG,
    "topic_modeling": TOPIC_MODELING_CONFIG,
    "keyword": KEYWORD_CONFIG,
    "scraping": SCRAPING_CONFIG,
    "data": DATA_CONFIG,
    "visualization": VISUALIZATION_CONFIG,
    "logging": LOGGING_CONFIG,
    "model_storage": MODEL_STORAGE,
    "api": API_CONFIG,
    "cache": CACHE_CONFIG,
} 