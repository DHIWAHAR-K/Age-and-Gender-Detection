import pandas as pd
from pathlib import Path

def load_utkface_dataset(dataset_path):
    path = Path(dataset_path)
    filenames = list(map(lambda x: x.name, path.glob('*.jpg')))

    age_labels, gender_labels, image_path = [], [], []
    for filename in filenames:
        image_path.append(filename)
        temp = filename.split('_')
        age_labels.append(temp[0])
        gender_labels.append(temp[1])

    df = pd.DataFrame()
    df['image'], df['age'], df['gender'] = image_path, age_labels, gender_labels
    df = df.astype({'age':'float32', 'gender': 'int32'})
    return df 