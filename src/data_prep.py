import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml  # For params

# Load params
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Load original dataset.csv (add if missing: echo "feature1,feature2,label\n1,2,0\n3,4,1\n5,6,0" > dataset.csv)
df = pd.read_csv('dataset.csv')

# Synthetic data generation (as before)
np.random.seed(42)
n_samples = 1000
feature1 = np.random.uniform(0, 10, n_samples)
feature2 = feature1 * 0.8 + np.random.normal(0, 1, n_samples)
labels = ((feature1 + feature2 > 5).astype(int))
synthetic_df = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'label': labels})
df = pd.concat([df, synthetic_df], ignore_index=True).drop_duplicates()

# Split using params
X = df[['feature1', 'feature2']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['train']['test_size'], random_state=42, stratify=y)

# Save to data/
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f"Train: {train_df.shape}, Test: {test_df.shape}")