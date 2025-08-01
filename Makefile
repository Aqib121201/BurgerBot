# BürgerBot Makefile
# Common commands for project management

.PHONY: help install test clean run-dashboard run-pipeline docker-build docker-run

# Default target
help:
	@echo "🇩🇪 BürgerBot - German Government Document Analysis System"
	@echo ""
	@echo "Available commands:"
	@echo "  install         Install dependencies"
	@echo "  test           Run unit tests"
	@echo "  clean          Clean temporary files"
	@echo "  run-dashboard  Start Streamlit dashboard"
	@echo "  run-pipeline   Run complete NLP pipeline"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo "  setup          Complete project setup"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Run tests
test:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests completed!"

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "✅ Cleanup completed!"

# Start Streamlit dashboard
run-dashboard:
	@echo "🚀 Starting Streamlit dashboard..."
	streamlit run app/app.py

# Run complete pipeline
run-pipeline:
	@echo "🔄 Running complete NLP pipeline..."
	python run_pipeline.py --full

# Run data preprocessing only
preprocess:
	@echo "📄 Running data preprocessing..."
	python run_pipeline.py --preprocess --scrape-pdfs --process-pdfs --create-chunks

# Run model training only
train:
	@echo "🧠 Running model training..."
	python run_pipeline.py --train

# Run visualization only
visualize:
	@echo "📊 Running visualization pipeline..."
	python run_pipeline.py --visualize

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -f docker/Dockerfile -t burgerbot .
	@echo "✅ Docker image built successfully!"

# Run Docker container
docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8501:8501 burgerbot

# Run Docker with pipeline execution
docker-run-pipeline:
	@echo "🐳 Running Docker container with pipeline execution..."
	docker run -p 8501:8501 -e RUN_PIPELINE=true burgerbot

# Complete project setup
setup: install
	@echo "🔧 Setting up project directories..."
	mkdir -p data/{raw,processed,external}
	mkdir -p models visualizations logs configs
	@echo "✅ Project setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Add German government PDFs to data/raw/"
	@echo "2. Run: make preprocess"
	@echo "3. Run: make train"
	@echo "4. Run: make run-dashboard"

# Development setup
dev-setup: setup
	@echo "🔧 Setting up development environment..."
	pip install pytest pytest-cov black flake8
	@echo "✅ Development setup completed!"

# Format code
format:
	@echo "🎨 Formatting code..."
	black src/ tests/ app/
	@echo "✅ Code formatting completed!"

# Lint code
lint:
	@echo "🔍 Linting code..."
	flake8 src/ tests/ app/
	@echo "✅ Code linting completed!"

# Run with coverage
test-coverage:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ --cov=src --cov-report=html
	@echo "✅ Coverage report generated!"

# Create sample data
sample-data:
	@echo "📄 Creating sample data..."
	mkdir -p data/raw
	@echo "Sample PDF files should be placed in data/raw/"
	@echo "You can download German government PDFs from:"
	@echo "- https://www.bundestag.de"
	@echo "- https://www.bundesregierung.de"
	@echo "- https://www.bundesrat.de"

# Show project status
status:
	@echo "📊 Project Status:"
	@echo "Python version: $(shell python --version)"
	@echo "Dependencies: $(shell pip list | wc -l) packages installed"
	@echo "Data files: $(shell find data/ -name "*.pdf" | wc -l) PDFs"
	@echo "Models: $(shell find models/ -name "*.joblib" -o -name "*.pkl" | wc -l) saved models"
	@echo "Visualizations: $(shell find visualizations/ -name "*.png" | wc -l) plots"

# Quick demo
demo:
	@echo "🎬 Running quick demo..."
	@echo "This will run a small sample through the pipeline"
	python run_pipeline.py --dry-run
	@echo "Demo completed! Run 'make run-dashboard' to see the interface." 