import numpy as np
from visualize import plot_loss
from preprocess import prepare_images
from model_builder import build_model
from data_loader import load_utkface_dataset
from sklearn.model_selection import train_test_split

# Constants
DATASET_PATH = "./UTKFace"
IMG_SIZE = (128, 128)
EPOCHS = 20
BATCH_SIZE = 10

# Load and preprocess
df = load_utkface_dataset(DATASET_PATH)
df = df.sample(frac=1, random_state=10)  # shuffle
train_df, _ = train_test_split(df, test_size=0.85, random_state=42)

x_train, y_gender, y_age = prepare_images(train_df, DATASET_PATH, IMG_SIZE)

# Build & Train model
model = build_model(input_size=(IMG_SIZE[0], IMG_SIZE[1], 1))
model.summary()

history = model.fit(x_train, [y_gender, y_age], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

# Save graphs
plot_loss(history)