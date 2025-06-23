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

Dataset Structure
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


