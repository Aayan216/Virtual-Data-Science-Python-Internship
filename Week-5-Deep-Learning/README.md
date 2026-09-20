# Week 5 — Deep Learning

## Project: Fashion-MNIST Image Classification Using Convolutional Neural Networks

This project applies deep learning techniques to image classification using the **Fashion-MNIST** dataset. A baseline Multilayer Perceptron (MLP) and multiple Convolutional Neural Network (CNN) experiments were developed and evaluated using TensorFlow/Keras.

### Objective

Build and evaluate a neural-network-based image classification system that can classify 28×28 grayscale Fashion-MNIST images into 10 clothing categories. The project also investigates model design choices, augmentation, overfitting, and evaluation performance.

### Dataset

**Fashion-MNIST** contains 70,000 grayscale images across 10 classes:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The dataset is loaded through TensorFlow/Keras.

### Models and Experiments

The project includes:

1. **Baseline MLP** — used as a reference model.
2. **CNN without augmentation** — selected as the final model.
3. **CNN with heavy augmentation** — evaluated as a controlled experiment.
4. **CNN with light augmentation** — evaluated as a controlled experiment.

The final CNN uses convolutional layers with Batch Normalization, ReLU activation, Max Pooling, Dropout, Global Average Pooling, and fully connected classification layers.

### Final Model Performance

The selected **CNN without augmentation** achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **91.91%** |
| Precision | **92.00%** |
| Recall | **91.91%** |
| F1-Score | **91.93%** |
| Test Loss | **0.2287** |
| Correct Predictions | **9,191 / 10,000** |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Baseline MLP | 89.29% | 89.36% | 89.29% | 89.31% |
| CNN — No Augmentation | **91.91%** | **92.00%** | **91.91%** | **91.93%** |
| CNN — Heavy Augmentation | 87.03% | 88.23% | 87.03% | 87.30% |
| CNN — Light Augmentation | 82.84% | 83.44% | 82.84% | 82.42% |

### Key Findings

- The CNN without augmentation improved test accuracy over the baseline MLP.
- The final model correctly classified **9,191 of 10,000** test images.
- **Bag, Sneaker, Trouser, and Ankle boot** had particularly strong class-level performance.
- **Shirt** was the most difficult class, with **78.8%** class accuracy.
- The main errors occurred between visually similar upper-body clothing classes, especially Shirt, T-shirt/top, Coat, and Pullover.
- The augmentation experiments were included to study how augmentation strength affected generalization; in these experiments, the augmented models performed below the non-augmented CNN.

### Files

- `Yuva_Week5_Fashion_MNIST_Deep_Learning.ipynb` — complete implementation, experiments, training, and evaluation.
- `Yuva_Week5_Fashion_MNIST_Deep_Learning_Report.docx` — detailed internship report.
- `fashion_mnist_final_results.zip` — archived final experiment outputs and evaluation results.

### Tools and Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

### Project Workflow

**Dataset Loading → Preprocessing → Baseline MLP → CNN Design → Training → Evaluation → Augmentation Experiments → Error Analysis → Final Model Selection**

