# African Language Models Project

This project implements custom language models for African languages using a hybrid approach of local development and cloud computing.

## Project Structure

```
.
├── src/year1/          # Main source code
│   ├── main.py         # Main application
│   └── utils/          # Utility functions
├── tests/              # Test files
├── data/               # Data files
├── scripts/            # Development scripts
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project configuration
└── README.md          # This file
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

## Development Workflow

### Running the Application
```bash
python -m src.year1.main
```

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/ tests/
flake8 src/ tests/
```

## Data Science Stack

This project uses the following libraries:

- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis  
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning algorithms

## Project Goals

1. Build custom language models for African languages
2. Implement data processing pipelines
3. Create visualization tools for model analysis
4. Develop testing frameworks for model validation

## Getting Started

1. Clone the repository
2. Set up the virtual environment
3. Install dependencies
4. Run the main application
5. Explore the test suite