# LSTM Text Similarity on STSB Dataset

This project aims to evaluate text similarity using an LSTM-based model trained on the STSB (Semantic Textual Similarity Benchmark) dataset. The project utilizes a Bi-LSTM architecture along with BERT tokenization to predict the similarity score between pairs of sentences.

## Project Structure

- **src/**: Contains the core code for the project.
  - **model.py**: Defines the LSTM-based model architecture used for predicting text similarity.
  - **preprocess.py**: Includes functions for preparing and tokenizing the dataset.
  - **utils.py**: Utility functions for training, evaluating, and plotting metrics.

- **evaluate_stsb_metrics.py**: A script for evaluating the model on the STSB test set. It calculates various metrics including Pearson Correlation, Mean Squared Error (MSE), R² Score, and Average Cosine Similarity.

- **predict_stsb_similarity.py**: A script for predicting similarity scores between two user-provided sentences using the trained LSTM model.

- **data/**: Contains raw and preprocessed data for training and evaluation.
  - **processed/**: Contains processed versions of the dataset.
  - **raw/**: Contains the raw dataset files.
    - **huggingface/**: Raw data from Hugging Face.
    - **kaggle/**: Raw data from Kaggle.

- **notebooks/**: Jupyter notebooks used for exploratory data analysis and prototyping.

- **outputs/**: Stores model weights, evaluation metrics, and plots generated during training and evaluation.

## Setup Instructions

### Prerequisites

- Python >= 3.7
- PyTorch >= 1.7
- Transformers
- Datasets (Hugging Face)
- Scikit-learn
- Tqdm

You can install the required Python packages by running:

```sh
pip install -r requirements.txt
```

### Data Preparation

The STSB dataset should be available in the `data/raw/kaggle/` directory, with files named:

- `stsb_train.tsv`
- `stsb_dev.tsv`
- `stsb_test.tsv`

### Running the Code

#### 1. Train the Model

Currently, this project uses an LSTM model which is already trained. If you want to train from scratch, you need to create a training script similar to `train.py`. Model parameters are saved in the `outputs/` folder.

#### 2. Evaluate the Model

To evaluate the model on the STSB test dataset, run the following command:

```sh
python evaluate.py
```

This script will calculate and print metrics like Pearson Correlation, MSE, R² Score, and Average Cosine Similarity, and save the results in the `outputs/` folder.

#### 3. Predict Sentence Similarity

You can use the `predict.py` script to predict the similarity between two user-provided sentences:

```sh
python predict.py
```

The script will prompt you to enter two sentences, and then it will output the predicted similarity score.

## Model Details

- **Architecture**: Bi-LSTM with 2 layers, trained using BERT embeddings.
- **Hyperparameters**:
  - **Embedding Dimension**: 1536
  - **Hidden Dimension**: 768
  - **Number of Layers**: 2
  - **Bidirectional**: True
  - **Dropout**: 0.2
  - **Batch Size**: 512
  - **Learning Rate**: 0.0005
- **Loss Function**: MSE (Mean Squared Error) to train the model to predict similarity scores ranging from 0 to 1.

## Evaluation Metrics

- **Best Validation Loss**: 0.1078
- **Final Training Loss**: 0.0834
- **Final Training Accuracy**: 0.927
- **Final Validation Loss**: 0.1171
- **Final Validation Accuracy**: 0.836
- **Final Test Loss**: 0.0948
- **Final Test Accuracy**: 0.900
- **Pearson Correlation**: 0.180

- **Pearson Correlation**: Measures linear correlation between predictions and true scores.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted values and actual values.
- **R² Score**: Represents the proportion of variance in the dependent variable that is predictable from the independent variable.
- **Average Cosine Similarity**: Measures similarity between embeddings produced by the LSTM for the two input sentences.

## Future Work

- Improve the model architecture by incorporating a transformer-based model like BERT or RoBERTa.
- Experiment with different loss functions such as Huber Loss or Cosine Embedding Loss to improve prediction quality.
- Add additional regularization methods like L2 regularization to improve model generalization.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

- The project utilizes the STSB dataset for training and evaluation.
- Thanks to the Hugging Face community for providing helpful tools and pre-trained models.

If you have any questions or suggestions, feel free to reach out!

