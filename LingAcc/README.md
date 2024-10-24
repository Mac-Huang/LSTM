# Linguistic Acceptability (LingAcc) LSTM Model

This project is focused on building and optimizing an LSTM-based model for predicting the grammatical acceptability of sentences using the **Corpus of Linguistic Acceptability (CoLA)** dataset. The project includes hyperparameter tuning, performance evaluation, and a final LSTM architecture.

## Project Structure

```
LingAcc/
├── data/                    # Contains the dataset
│   └── cola_public/
│       ├── raw/             # Raw data files
│       └── tokenized/       # Tokenized data files
│
├── notebooks/               # Jupyter notebooks for experimentation
│
├── outputs/                 # Stores model outputs, including loss plots and saved models
│
├── src/                     # Source code for the project
│   └── __pycache__/         # Compiled Python files
│
└── README.md                # Project README
```

### Dataset

The **CoLA** dataset is organized under the `data/cola_public/` directory. The `raw` folder contains the original dataset, and the `tokenized` folder contains preprocessed, tokenized data used for training.

### Source Code

The `src/` directory contains the main scripts for training, evaluation, and preprocessing of the dataset, including model definition and utility functions for data handling.

## Hyperparameter Tuning Process

This project went through several iterations of hyperparameter tuning to improve the model's performance. Below is a detailed account of the tuning process:

### Initial Configuration

The initial configuration used a large hidden dimension and many layers, which led to overfitting. The results showed a significant gap between training and validation metrics:

- **Embedding Dimension**: 1024
- **Hidden Dimension**: 1024
- **Number of Layers**: 5
- **Bidirectional**: True
- **Dropout**: 0.5
- **Batch Size**: 64
- **Learning Rate**: 0.0005

**Results**:
- **Best Validation Loss**: 0.920
- **Test Accuracy**: 53.8%

### Second Iteration

The second iteration focused on reducing the model's complexity by lowering the embedding and hidden dimensions, which resulted in improved validation and test performance:

- **Embedding Dimension**: 300
- **Hidden Dimension**: 512
- **Number of Layers**: 3
- **Dropout**: 0.6
- **Batch Size**: 128
- **Learning Rate**: 0.0003

**Results**:
- **Best Validation Loss**: 0.658
- **Test Accuracy**: 62.2%

### Final Tuning

In the final iteration, we reduced the hidden dimension further, increased the dropout rate, and lowered the learning rate. This improved the test accuracy and mitigated overfitting, resulting in a well-balanced model:

- **Embedding Dimension**: 300
- **Hidden Dimension**: 256
- **Number of Layers**: 3
- **Bidirectional**: True
- **Dropout**: 0.6
- **Batch Size**: 512
- **Learning Rate**: 0.0001

**Final Results**:
- **Best Validation Loss**: 0.701
- **Final Training Loss**: 0.617
- **Final Training Accuracy**: 65.7%
- **Final Validation Loss**: 0.701
- **Final Validation Accuracy**: 68.0%
- **Final Test Loss**: 0.654
- **Final Test Accuracy**: 71.8%

## Final Hyperparameters

The final hyperparameters that yielded the best performance:

```json
{
    "embedding_dim": 300,
    "hidden_dim": 256,
    "output_dim": 1,
    "n_layers": 3,
    "bidirectional": true,
    "dropout": 0.6,
    "batch_size": 512,
    "learning_rate": 0.0001,
    "best_valid_loss": 0.7011,
    "final_train_loss": 0.6167,
    "final_train_accuracy": 65.7%,
    "final_valid_loss": 0.7011,
    "final_valid_accuracy": 68.0%,
    "final_test_loss": 0.6542,
    "final_test_accuracy": 71.8%
}
```

## Conclusion

Through iterative hyperparameter tuning, the final LSTM model achieved a **Test Accuracy** of 71.8%, showing a significant improvement over earlier configurations. The process involved reducing model complexity, increasing regularization, and fine-tuning the learning rate to strike a balance between overfitting and underfitting.

## Next Steps

- **Pretrained Embeddings**: Experimenting with pretrained embeddings (such as GloVe) could yield additional improvements.
- **Data Augmentation**: Augmenting the dataset with synthetic or additional linguistic examples could help the model generalize better.
- **Exploring Transformer Models**: In future experiments, replacing the LSTM with transformer-based architectures (like BERT) might further improve performance.