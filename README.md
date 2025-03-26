# Age and Gender Detection
This project builds a Convolutional Neural Network (CNN) model that simultaneously predicts the **gender** (classification) and **age** (regression) of individuals from facial images in the [UTKFace dataset](https://susanqq.github.io/UTKFace/). It uses grayscale images and leverages a multi-output deep learning model.

## Project Structure:

- `data_loader.py`: Loads image file names and extracts age and gender labels from filenames.
- `preprocess.py`: Preprocesses images and prepares input arrays for model training.
- `model_builder.py`: Defines and compiles the CNN model with dual outputs for gender and age.
- `visualize.py`: Plots and saves training and validation loss graphs.
- `main.py`: Main training script – handles data loading, preprocessing, model training, and visualization.
- `predict.py`: Contains a function to test the model on a single sample and visualize the result.

## Setup and Installation:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/utkface-age-gender-prediction.git
cd utkface-age-gender-prediction
```

### 2. Install Required Packages

Make sure you have Python 3.6+ and install dependencies using pip:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 3. Download Dataset

Create a directory called 'data' and then Download the [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) Dataset and extract it into the data directory:

```bash
data/UTKFACE
```


## Usage:

Train the model

```bash
python main.py
```

This will:

1. Load and shuffle the dataset

2. Preprocess 15% of the data (for faster training)

3. Train a CNN model for EPOCHS (default: 20)

4. Save training loss graphs into the graphs/ folder


## Predict on a Sample Image:

Modify and run predict.py

```bash
predict_and_display(model, x_train, y_gender, y_age, index=0)
```


This will:

1. Print actual vs. predicted age and gender

2. Display the image with matplotlib


## Model Architecture:

Input: 128×128 grayscale facial image

Shared Convolutional Layers:

Conv2D → BatchNorm → MaxPooling (x3)

Gender Branch:

Dense → Dropout → Sigmoid Output

Age Branch:

Dense → Dense → Dropout → ReLU Output

## Loss Functions:

Gender: Binary Cross-Entropy

Age: Mean Squared Error (MSE)

## License:

This project is intended for educational and research purposes. Feel free to fork and modify for your use.
