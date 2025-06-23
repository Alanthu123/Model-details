# Multi-Modal Terrain Classification

This repository contains a PyTorch implementation of a multi-modal neural network for terrain classification using both image data and sequence data. The model classifies terrain into five categories: Concrete, Sand, Grass, Stone, and Water.

## Features
- **Dual-input CNN architecture**: Processes image and sequence data simultaneously
- **Dynamic feature weighting**: Adjusts contribution of image vs. sequence features
- **Mixed precision training**: Accelerates training using CUDA AMP
- **Comprehensive evaluation**: Includes confusion matrices and loss visualization

## Requirements
- Python 3.7+
- PyTorch 1.10+
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Pillow

Install dependencies:
```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn pillow
```

## Dataset Structure
Organize your dataset in the following structure:

```bash
dataset_root/
├── Concrete/
│   ├── seq_1.xlsx
│   ├── img_1.jpg
│   └── ...
├── Sand/
├── Grass/
├── Stone/
└── Water/
```

## Usage
Update the data_dir path in the script to point to your dataset
Adjust hyperparameters as needed (image/sequence weights, sliding window size)
Run the training script:
```bash
python TrainAndTest.py
```

## Key Components
### Data Loading
Processes Excel files containing sequence data (4 columns x sliding_window rows). Loads corresponding JPG images (resized to 128x128). Applies normalization to both data types.

### Model Architecture
Image Pathway:
```bash
Conv2d(3,16) → ReLU → MaxPool → Conv2d(16,32) → ReLU → MaxPool → Flatten
```

Sequence Pathway:
```bash
Conv1d(4,8) → ReLU → Conv1d(8,16) → ReLU → Flatten
```
Combined:
```bash
[Image Features * weight] + [Sequence Features * weight] → FC(64) → Output(5)
```
### Training
200 epochs with early stopping
Adam optimizer (lr=0.001)
Cross-entropy loss
Automatic Mixed Precision (AMP)

### Evaluation
Confusion matrix (raw counts and normalized)
Excel export of classification results
Training/validation loss curves

## Results Visualization
The script automatically generates:
1. Training/validation loss curves
2. Dynamic weight adjustment plots
3. Validation accuracy progression
4. Normalized confusion matrix
5. Weight-accuracy analysis curves

## Customization
Modify MultiModalModel class for architectural changes
Adjust load_data() parameters for different:
1. Sliding window sizes
2. Maximum samples per class
3. Image dimensions
4. Tune hyperparameters in training section
