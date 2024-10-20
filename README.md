# LSTM Project

This is the main branch of the LSTM project repository. It contains the core components that are shared across different tasks, including the base LSTM model and reusable utility functions.

## Overview
The main branch provides the general-purpose LSTM model implementation and utility functions. These components serve as the foundation for various tasks like sentiment analysis and text similarity, which are implemented in dedicated branches.

## Branches
This repository includes multiple branches, each serving a distinct purpose:

- **Main Branch** (current branch): Contains the shared LSTM model (`model.py`) and general utility scripts (`utils.py`). This branch is intended to act as the base for other tasks.
- **SentiAnaly**: This branch contains all the scripts and configurations required for training and evaluating an LSTM model for sentiment analysis.
- **TextSimi**: This branch focuses on implementing text similarity calculations using the LSTM model. It includes training scripts and configurations for fine-tuning the LSTM on similarity tasks like STSB (Semantic Textual Similarity Benchmark).

## Directory Structure
The structure of this main branch is as follows:

```
LSTM/
├── src/
│   ├── model.py                # Base LSTM model code (shared)
│   ├── utils.py                # Shared utility functions (e.g., accuracy calculation, evaluation)
│   └── __init__.py             # Makes 'src' a package
├── data/
│   ├── row_data/               # Raw dataset (used for both sentiment and similarity tasks)
│   ├── clean_data/             # Cleaned data that can be shared across tasks
│   └── stopwords/              # Stopwords for data preprocessing
├── notebooks/                  # Jupyter notebooks for exploratory analysis
├── README.md                   # General README for the main branch
└── requirements.txt            # Python dependencies
```

## How to Use

### Setting Up the Environment
To set up the environment and install the required dependencies, use the following commands:

```sh
# Clone the repository
git clone <repository-url>

# Navigate to the repository
cd LSTM

# Install the dependencies
pip install -r requirements.txt
```

### Switching to Other Branches
To work on a specific task, switch to the appropriate branch:

- **Sentiment Analysis**: 
  ```sh
  git checkout SentiAnaly
  ```

- **Text Similarity**:
  ```sh
  git checkout TextSimi
  ```

Each branch contains its own README file with detailed instructions on how to use the scripts and configurations specific to that task.

## Contribution Guidelines
If you want to contribute to the project, please follow these guidelines:

1. Fork the repository and create your feature branch (`git checkout -b feature/new-feature`).
2. Commit your changes (`git commit -m 'Add some feature'`).
3. Push to the branch (`git push origin feature/new-feature`).
4. Open a pull request.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contact
If you have any questions or suggestions regarding the project, feel free to open an issue or contact the maintainers.

