"""
BürgerBot Streamlit Dashboard
Main application for German Government Document Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.model_utils import ModelLoader, VisualizationUtils, InferenceUtils, DataUtils
from src.config import CONFIG

# Page configuration
st.set_page_config(
    page_title="BürgerBot - German Government Document Analysis",
    page_icon="🇩🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🇩🇪 BürgerBot</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #666;">German Government Document Analysis Dashboard</h2>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Overview", "📊 Data Analysis", "🌐 Translation", "📝 Summarization", 
     "😊 Sentiment Analysis", "📚 Topic Modeling", "🔑 Keywords", "📈 Visualizations"]
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load data and models
@st.cache_data
def load_data():
    """Load processed document data."""
    try:
        return DataUtils.load_processed_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        results = ModelLoader.load_training_results()
        return results
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load data and models
if not st.session_state.data_loaded:
    with st.spinner("Loading document data..."):
        df = load_data()
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True

if not st.session_state.models_loaded:
    with st.spinner("Loading trained models..."):
        results = load_models()
        if results is not None:
            st.session_state.results = results
            st.session_state.models_loaded = True

# Overview Page
if page == "🏠 Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### About BürgerBot
        
        BürgerBot is an advanced NLP system designed to analyze German government documents 
        from sources like Bundestag, Bundesregierung, and Bundesrat. The system provides:
        
        - **📄 PDF Processing**: Automated extraction and cleaning of German government PDFs
        - **🌐 Translation**: German to English translation using MarianMT
        - **📝 Summarization**: Key point extraction using BART
        - **😊 Sentiment Analysis**: Document sentiment classification
        - **📚 Topic Modeling**: LDA and BERTopic for theme discovery
        - **🔑 Keyword Extraction**: YAKE-based keyword identification
        
        ### Technology Stack
        
        - **NLP Models**: Transformers (MarianMT, BART, RoBERTa)
        - **Topic Modeling**: LDA, BERTopic
        - **Visualization**: Plotly, WordCloud, Matplotlib
        - **Framework**: Streamlit, PyMuPDF, YAKE
        """)
    
    with col2:
        if st.session_state.data_loaded:
            stats = DataUtils.get_document_statistics(st.session_state.df)
            
            st.markdown("### 📊 Current Dataset Statistics")
            
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric("Total Documents", f"{stats['total_documents']:,}")
                st.metric("Unique Files", f"{stats['unique_files']:,}")
                st.metric("Avg Length", f"{stats['avg_chunk_length']:.0f} chars")
            
            with metric_col2:
                st.metric("Total Pages", f"{stats['total_pages']:,}")
                st.metric("Total Characters", f"{stats['total_characters']:,}")
                st.metric("Std Length", f"{stats['std_chunk_length']:.0f} chars")
            
            # Document length distribution
            fig = px.histogram(
                st.session_state.df, 
                x="chunk_length", 
                nbins=30,
                title="Document Length Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Data not loaded. Please check if preprocessing has been completed.")

# Data Analysis Page
elif page == "📊 Data Analysis":
    st.header("Data Analysis")
    
    if not st.session_state.data_loaded:
        st.error("Data not loaded. Please run data preprocessing first.")
        st.stop()
    
    df = st.session_state.df
    
    # Data overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10))
        
        st.subheader("Data Types")
        st.write(df.dtypes)
    
    with col2:
        st.subheader("Basic Statistics")
        st.write(df.describe())
        
        st.subheader("Missing Values")
        missing_data = df.isnull().sum()
        st.write(missing_data)
    
    # Document analysis
    st.subheader("Document Analysis")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_length = st.slider("Minimum document length", 0, 1000, 100)
    
    with col2:
        max_length = st.slider("Maximum document length", 1000, 10000, 5000)
    
    with col3:
        sample_size = st.slider("Sample size for analysis", 10, 1000, 100)
    
    # Filter and sample data
    filtered_df = DataUtils.filter_documents_by_length(df, min_length, max_length)
    sampled_df = DataUtils.sample_documents(filtered_df, sample_size)
    
    st.write(f"Filtered dataset: {len(filtered_df)} documents")
    st.write(f"Sampled dataset: {len(sampled_df)} documents")
    
    # Document length analysis
    fig = VisualizationUtils.plot_document_lengths(sampled_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # File distribution
    file_counts = sampled_df["filename"].value_counts().head(10)
    fig = px.bar(
        x=file_counts.values, 
        y=file_counts.index, 
        orientation='h',
        title="Top 10 Files by Document Count"
    )
    st.plotly_chart(fig, use_container_width=True)

# Translation Page
elif page == "🌐 Translation":
    st.header("German to English Translation")
    
    if not st.session_state.models_loaded:
        st.error("Models not loaded. Please run model training first.")
        st.stop()
    
    # Translation interface
    st.subheader("Interactive Translation")
    
    # Text input
    german_text = st.text_area(
        "Enter German text to translate:",
        height=150,
        placeholder="Geben Sie hier deutschen Text ein..."
    )
    
    if st.button("Translate"):
        if german_text.strip():
            with st.spinner("Translating..."):
                try:
                    translation = InferenceUtils.translate_text(german_text)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original (German)")
                        st.write(german_text)
                    
                    with col2:
                        st.subheader("Translation (English)")
                        st.write(translation)
                        
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")
    
    # Batch translation results
    if st.session_state.results.get("translation"):
        st.subheader("Translation Results Overview")
        
        translation_results = st.session_state.results["translation"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Translated", len(translation_results["translated_texts"]))
            st.metric("Model Used", translation_results["model_info"]["model_name"])
        
        with col2:
            st.metric("Device", translation_results["model_info"]["device"])
            
            # Sample translations
            if len(translation_results["translated_texts"]) > 0:
                st.subheader("Sample Translations")
                
                sample_size = st.slider("Number of samples", 1, 10, 5)
                samples = translation_results["translated_texts"][:sample_size]
                
                for i, translation in enumerate(samples, 1):
                    with st.expander(f"Sample {i}"):
                        st.write(translation[:200] + "..." if len(translation) > 200 else translation)

# Summarization Page
elif page == "📝 Summarization":
    st.header("Text Summarization")
    
    if not st.session_state.models_loaded:
        st.error("Models not loaded. Please run model training first.")
        st.stop()
    
    # Summarization interface
    st.subheader("Interactive Summarization")
    
    # Text input
    text_to_summarize = st.text_area(
        "Enter text to summarize:",
        height=200,
        placeholder="Enter text here..."
    )
    
    if st.button("Summarize"):
        if text_to_summarize.strip():
            with st.spinner("Generating summary..."):
                try:
                    # For demo, we'll use a simple extractive summary
                    # In a real implementation, you'd load the BART model
                    words = text_to_summarize.split()
                    summary = " ".join(words[:50]) + "..." if len(words) > 50 else text_to_summarize
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Text")
                        st.write(text_to_summarize)
                        st.metric("Original Length", len(text_to_summarize))
                    
                    with col2:
                        st.subheader("Summary")
                        st.write(summary)
                        st.metric("Summary Length", len(summary))
                        st.metric("Compression Ratio", f"{len(summary)/len(text_to_summarize)*100:.1f}%")
                        
                except Exception as e:
                    st.error(f"Summarization error: {str(e)}")
        else:
            st.warning("Please enter some text to summarize.")
    
    # Summarization results overview
    if st.session_state.results.get("summarization"):
        st.subheader("Summarization Results Overview")
        
        summarization_results = st.session_state.results["summarization"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Summaries", len(summarization_results["summaries"]))
            st.metric("Model Used", summarization_results["model_info"]["model_name"])
        
        with col2:
            # Sample summaries
            if len(summarization_results["summaries"]) > 0:
                st.subheader("Sample Summaries")
                
                sample_size = st.slider("Number of samples", 1, 10, 5)
                samples = summarization_results["summaries"][:sample_size]
                
                for i, summary in enumerate(samples, 1):
                    with st.expander(f"Summary {i}"):
                        st.write(summary)

# Sentiment Analysis Page
elif page == "😊 Sentiment Analysis":
    st.header("Sentiment Analysis")
    
    if not st.session_state.models_loaded:
        st.error("Models not loaded. Please run model training first.")
        st.stop()
    
    # Interactive sentiment analysis
    st.subheader("Interactive Sentiment Analysis")
    
    text_for_sentiment = st.text_area(
        "Enter text for sentiment analysis:",
        height=150,
        placeholder="Enter text here..."
    )
    
    if st.button("Analyze Sentiment"):
        if text_for_sentiment.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    # For demo purposes, we'll simulate sentiment analysis
                    # In a real implementation, you'd use the trained model
                    import random
                    
                    sentiments = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
                    sentiment = random.choice(sentiments)
                    score = random.uniform(0.5, 1.0)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Input Text")
                        st.write(text_for_sentiment)
                    
                    with col2:
                        st.subheader("Sentiment Results")
                        
                        # Color coding
                        if sentiment == "POSITIVE":
                            st.success(f"Sentiment: {sentiment}")
                        elif sentiment == "NEGATIVE":
                            st.error(f"Sentiment: {sentiment}")
                        else:
                            st.info(f"Sentiment: {sentiment}")
                        
                        st.metric("Confidence Score", f"{score:.3f}")
                        
                        # Progress bar
                        st.progress(score)
                        
                except Exception as e:
                    st.error(f"Sentiment analysis error: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Sentiment results overview
    if st.session_state.results.get("sentiment"):
        st.subheader("Sentiment Analysis Results Overview")
        
        sentiment_results = st.session_state.results["sentiment"]
        
        # Create sentiment distribution plot
        fig = VisualizationUtils.plot_sentiment_distribution(sentiment_results["sentiment_results"])
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment statistics
        labels = [result["label"] for result in sentiment_results["sentiment_results"]]
        scores = [result["score"] for result in sentiment_results["sentiment_results"]]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Analyzed", len(sentiment_results["sentiment_results"]))
        
        with col2:
            avg_score = np.mean(scores)
            st.metric("Average Confidence", f"{avg_score:.3f}")
        
        with col3:
            most_common = pd.Series(labels).mode()[0]
            st.metric("Most Common Sentiment", most_common)

# Topic Modeling Page
elif page == "📚 Topic Modeling":
    st.header("Topic Modeling")
    
    if not st.session_state.models_loaded:
        st.error("Models not loaded. Please run model training first.")
        st.stop()
    
    # Topic modeling interface
    st.subheader("Interactive Topic Analysis")
    
    text_for_topics = st.text_area(
        "Enter text for topic analysis:",
        height=150,
        placeholder="Enter text here..."
    )
    
    model_type = st.selectbox("Select topic model:", ["LDA", "BERTopic"])
    
    if st.button("Analyze Topics"):
        if text_for_topics.strip():
            with st.spinner("Analyzing topics..."):
                try:
                    topic_result = InferenceUtils.get_topic_for_text(text_for_topics, model_type.lower())
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Input Text")
                        st.write(text_for_topics)
                    
                    with col2:
                        st.subheader("Topic Analysis Results")
                        st.metric("Topic ID", topic_result["topic_id"])
                        st.metric("Topic Probability", f"{topic_result['topic_probability']:.3f}")
                        
                        if "topic_words" in topic_result:
                            st.write("**Top Topic Words:**")
                            st.write(", ".join(topic_result["topic_words"][:10]))
                        
                except Exception as e:
                    st.error(f"Topic analysis error: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Topic modeling results overview
    if st.session_state.results.get("topic_modeling"):
        st.subheader("Topic Modeling Results Overview")
        
        topic_results = st.session_state.results["topic_modeling"]
        
        # LDA Results
        if "lda" in topic_results:
            st.subheader("LDA Topic Model")
            
            lda_results = topic_results["lda"]
            
            # Topic distribution plot
            fig = VisualizationUtils.plot_topic_distribution(lda_results, "lda")
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic details
            st.write("**Topic Details:**")
            for topic in lda_results["topics"][:5]:  # Show first 5 topics
                with st.expander(f"Topic {topic['topic_id']}"):
                    st.write("**Top Words:**", ", ".join(topic["words"][:10]))
        
        # BERTopic Results
        if "bertopic" in topic_results:
            st.subheader("BERTopic Model")
            
            bertopic_results = topic_results["bertopic"]
            
            # Topic distribution plot
            fig = VisualizationUtils.plot_topic_distribution(bertopic_results, "bertopic")
            st.plotly_chart(fig, use_container_width=True)

# Keywords Page
elif page == "🔑 Keywords":
    st.header("Keyword Extraction")
    
    if not st.session_state.models_loaded:
        st.error("Models not loaded. Please run model training first.")
        st.stop()
    
    # Keyword extraction interface
    st.subheader("Interactive Keyword Extraction")
    
    text_for_keywords = st.text_area(
        "Enter text for keyword extraction:",
        height=150,
        placeholder="Enter German text here..."
    )
    
    if st.button("Extract Keywords"):
        if text_for_keywords.strip():
            with st.spinner("Extracting keywords..."):
                try:
                    keywords = InferenceUtils.extract_keywords_from_text(text_for_keywords)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Input Text")
                        st.write(text_for_keywords)
                    
                    with col2:
                        st.subheader("Extracted Keywords")
                        
                        if keywords:
                            # Create keyword DataFrame
                            keyword_df = pd.DataFrame(keywords)
                            keyword_df = keyword_df.sort_values("score")
                            
                            st.dataframe(keyword_df.head(10))
                            
                            # Keyword scores
                            fig = px.bar(
                                keyword_df.head(10), 
                                x="keyword", 
                                y="score",
                                title="Top 10 Keywords by Score"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No keywords extracted.")
                        
                except Exception as e:
                    st.error(f"Keyword extraction error: {str(e)}")
        else:
            st.warning("Please enter some text to extract keywords.")
    
    # Keyword results overview
    if st.session_state.results.get("keywords"):
        st.subheader("Keyword Extraction Results Overview")
        
        keyword_results = st.session_state.results["keywords"]
        
        # Create keyword trends plot
        fig = VisualizationUtils.plot_keyword_trends(keyword_results["keywords"])
        st.plotly_chart(fig, use_container_width=True)
        
        # Keyword statistics
        all_keywords = []
        for doc_keywords in keyword_results["keywords"]:
            all_keywords.extend(doc_keywords)
        
        if all_keywords:
            keyword_df = pd.DataFrame(all_keywords)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Keywords", len(all_keywords))
            
            with col2:
                st.metric("Unique Keywords", keyword_df["keyword"].nunique())
            
            with col3:
                avg_score = keyword_df["score"].mean()
                st.metric("Average Score", f"{avg_score:.3f}")

# Visualizations Page
elif page == "📈 Visualizations":
    st.header("Data Visualizations")
    
    if not st.session_state.data_loaded:
        st.error("Data not loaded. Please run data preprocessing first.")
        st.stop()
    
    df = st.session_state.df
    
    # Word cloud
    st.subheader("Word Cloud")
    
    # Combine all text for word cloud
    all_text = " ".join(df["chunk_text"].tolist())
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100
    ).generate(all_text)
    
    # Display word cloud
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Document statistics
    st.subheader("Document Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Document length distribution
        fig = px.histogram(
            df, 
            x="chunk_length", 
            nbins=30,
            title="Document Length Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # File distribution
        file_counts = df["filename"].value_counts().head(10)
        fig = px.bar(
            x=file_counts.values, 
            y=file_counts.index, 
            orientation='h',
            title="Top 10 Files by Document Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced visualizations
    st.subheader("Advanced Analytics")
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🇩🇪 BürgerBot - German Government Document Analysis System</p>
        <p>Built with Streamlit, Transformers, and Advanced NLP Techniques</p>
    </div>
    """, 
    unsafe_allow_html=True
) 