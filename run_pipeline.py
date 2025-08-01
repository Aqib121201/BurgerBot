#!/usr/bin/env python3
"""
BürgerBot Pipeline Orchestrator
Main CLI entry point for running the complete NLP pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_utils import DataUtils, VisualizationUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_data_preprocessing(args):
    """Run the data preprocessing pipeline."""
    logger.info("Starting data preprocessing pipeline...")
    
    try:
        preprocessor = DataPreprocessor()
        
        if args.scrape_pdfs:
            logger.info("Step 1: Scraping and downloading PDFs...")
            downloaded_files = preprocessor.scrape_and_download_pdfs()
            logger.info(f"Downloaded {len(downloaded_files)} PDF files")
        
        if args.process_pdfs:
            logger.info("Step 2: Processing PDFs...")
            df = preprocessor.process_pdf_directory(
                Path("data/raw"), 
                Path("data/processed")
            )
            logger.info(f"Processed {len(df)} document pages")
        
        if args.create_chunks:
            logger.info("Step 3: Creating document chunks...")
            df = DataUtils.load_processed_data()
            chunked_df = preprocessor.create_document_chunks(df)
            logger.info(f"Created {len(chunked_df)} document chunks")
        
        logger.info("Data preprocessing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        return False


def run_model_training(args):
    """Run the model training pipeline."""
    logger.info("Starting model training pipeline...")
    
    try:
        trainer = ModelTrainer()
        
        # Load data
        logger.info("Loading processed data...")
        df = trainer.load_processed_data()
        logger.info(f"Loaded {len(df)} documents")
        
        # Train models
        results = trainer.train_all_models(df)
        
        logger.info("Model training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False


def run_visualization(args):
    """Run the visualization pipeline."""
    logger.info("Starting visualization pipeline...")
    
    try:
        # Load data
        df = DataUtils.load_processed_data()
        
        # Create visualizations
        logger.info("Creating document length visualization...")
        fig = VisualizationUtils.plot_document_lengths(df)
        
        # Create word cloud
        logger.info("Creating word cloud...")
        all_text = " ".join(df["chunk_text"].tolist())
        wordcloud = VisualizationUtils.create_wordcloud(all_text, "German Government Documents")
        
        logger.info("Visualization pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return False


def run_full_pipeline(args):
    """Run the complete pipeline."""
    logger.info("Starting complete BürgerBot pipeline...")
    
    start_time = datetime.now()
    
    # Step 1: Data preprocessing
    if not run_data_preprocessing(args):
        logger.error("Pipeline failed at data preprocessing step")
        return False
    
    # Step 2: Model training
    if not run_model_training(args):
        logger.error("Pipeline failed at model training step")
        return False
    
    # Step 3: Visualization
    if not run_visualization(args):
        logger.error("Pipeline failed at visualization step")
        return False
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"Complete pipeline finished successfully in {duration}")
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BürgerBot - German Government Document Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --full
  
  # Run only data preprocessing
  python run_pipeline.py --preprocess --scrape-pdfs --process-pdfs --create-chunks
  
  # Run only model training
  python run_pipeline.py --train
  
  # Run only visualization
  python run_pipeline.py --visualize
        """
    )
    
    # Pipeline options
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Run the complete pipeline (preprocessing + training + visualization)"
    )
    
    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Run data preprocessing pipeline"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Run model training pipeline"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Run visualization pipeline"
    )
    
    # Data preprocessing options
    parser.add_argument(
        "--scrape-pdfs", 
        action="store_true",
        help="Scrape and download PDFs from government websites"
    )
    
    parser.add_argument(
        "--process-pdfs", 
        action="store_true",
        help="Process downloaded PDFs and extract text"
    )
    
    parser.add_argument(
        "--create-chunks", 
        action="store_true",
        help="Create document chunks for processing"
    )
    
    # General options
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without actually running"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Validate arguments
    if not any([args.full, args.preprocess, args.train, args.visualize]):
        parser.print_help()
        return 1
    
    # Dry run
    if args.dry_run:
        logger.info("DRY RUN - No actual processing will be performed")
        if args.full:
            logger.info("Would run: preprocessing + training + visualization")
        if args.preprocess:
            logger.info("Would run: data preprocessing")
        if args.train:
            logger.info("Would run: model training")
        if args.visualize:
            logger.info("Would run: visualization")
        return 0
    
    # Run pipeline
    try:
        if args.full:
            success = run_full_pipeline(args)
        elif args.preprocess:
            success = run_data_preprocessing(args)
        elif args.train:
            success = run_model_training(args)
        elif args.visualize:
            success = run_visualization(args)
        else:
            success = False
        
        if success:
            logger.info("Pipeline completed successfully!")
            return 0
        else:
            logger.error("Pipeline failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 