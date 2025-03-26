import os
import matplotlib.pyplot as plt

def plot_loss(history, output_dir="graphs"):
    os.makedirs(output_dir, exist_ok=True)

    # Gender Loss
    plt.plot(history.history['gender_output_loss'], label='Train')
    plt.plot(history.history['val_gender_output_loss'], label='Validation')
    plt.title('Gender Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/gender_loss.png")
    plt.clf()

    # Age Loss
    plt.plot(history.history['age_output_loss'], label='Train')
    plt.plot(history.history['val_age_output_loss'], label='Validation')
    plt.title('Age Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/age_loss.png")
    plt.clf()