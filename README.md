# Anomaly Detection Using Autoencoders

This repository provides a PyTorch-based implementation of an anomaly detection system for the MVTec AD dataset using convolutional autoencoders. The system trains an autoencoder for each category in the dataset to reconstruct images and detect anomalies by measuring reconstruction loss.

## Features

- **Dataset Handling**: Supports the MVTec Anomaly Detection (MVTec AD) dataset with automated loading and preprocessing.
- **Autoencoder Architecture**: Uses a pre-trained EfficientNet encoder with a custom decoder for high-quality image reconstruction.
- **Training Framework**: Sequential training of models for all categories in the dataset.
- **Anomaly Detection**: Computes reconstruction loss to classify images as normal or anomalous.
- **Pre-trained Model Support**: Allows loading of pre-trained models for individual categories.

## Requirements

- Python 3.7+
- PyTorch 1.10+
- OpenCV
- torchvision
- tqdm

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your_username/anomaly_detection_autoencoder.git
cd anomaly_detection_autoencoder
pip install -r requirements.txt
```

## Dataset Structure

Organize the dataset as follows:

```
A:\Anomaly Detection\Data\
    bottle\
        train\
            good\
                000.png
                001.png
                ...
        test\
            broken_large\
                000.png
                001.png
                ...
            broken_small\
                000.png
                001.png
                ...
            contamination\
                000.png
                001.png
                ...
        ground_truth\
            broken_large\
                000_mask.png
                001_mask.png
                ...
            broken_small\
                000_mask.png
                001_mask.png
                ...
            contamination\
                000_mask.png
                001_mask.png
                ...
    cable\
        ...
    capsule\
        ...
    carpet\
        ...
```

Update the `BASE_DIR` in the code to the root path of your dataset.

## Usage

### Training

Train autoencoders for all categories in the dataset:

```bash
python anomaly_detection_autoencoder.py
```

### Anomaly Detection

Use the trained models to classify a test image:

```python
from anomaly_detection_autoencoder import classify_image

# Replace with your test image path
test_image_path = r"A:\Anomaly Detection\Data\bottle\test\broken_small\000.png"
result = classify_image(test_image_path, BASE_DIR)
print(f"Final classification result: {result}")
```

## Code Explanation

### Key Components

1. **MVTecDataset**: Custom PyTorch Dataset class to load and preprocess images.
2. **Autoencoder**: Autoencoder architecture with EfficientNet as the encoder and a custom decoder.
3. **Training Pipeline**:
   - Loops through all dataset categories.
   - Trains a separate autoencoder for each category.
   - Saves the trained models for reuse.
4. **Anomaly Detection Pipeline**:
   - Loads pre-trained models.
   - Reconstructs the input image.
   - Measures reconstruction loss to detect anomalies.

### Parameters

- **Image Dimensions**: All images are resized to 224x224.
- **Training Hyperparameters**:
  - Batch size: 16
  - Epochs: 10
  - Learning rate: 1e-4
- **Anomaly Detection Threshold**: Reconstruction loss > 0.05 is considered anomalous.

## Output

- During training, models are saved as `{category}_autoencoder.pth`.
- Anomaly detection outputs the categories with detected anomalies, along with the corresponding reconstruction losses and mask folder locations.

## Results

Example output of anomaly detection:

```
Defects detected in the following categories:
Category: bottle, Loss: 0.0763, Mask Folder: A:\Anomaly Detection\Data\bottle\ground_truth\broken_large
Final classification result: {'bottle': {'loss': 0.0763, 'mask_folder': 'A:\\Anomaly Detection\\Data\\bottle\\ground_truth\\broken_large'}}
```

## Future Enhancements

- Add visualization of reconstructed images and anomaly masks.
- Improve thresholding method for anomaly classification.
- Extend support for other datasets.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [PyTorch Documentation](https://pytorch.org/docs/)

