# train_unsupervised_model.py (Enhanced with Comprehensive Visualizations and IndexError Fix)

import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = 'data'
MODEL_DIR = 'saved_models'
VIS_DIR = 'visualized_data'
DATA_PATH = os.path.join(DATA_DIR, 'featurized_block_data.csv')
SEQUENCE_LENGTH = 10
THRESHOLD_MULTIPLIER = 1.75 

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Visualization Functions ===

def visualize_data_exploration(df, features, vis_dir):
    """Generates and saves plots for initial data exploration."""
    logging.info("Generating data exploration visualizations...")
    
    # 1. Feature Distributions
    plt.figure(figsize=(16, 12))
    num_features = len(features)
    cols = 4
    rows = (num_features + cols - 1) // cols
    for i, col in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}', fontsize=10)
        plt.xlabel('')
        plt.ylabel('')
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(vis_dir, 'feature_distributions.png'))
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_correlation_heatmap.png'))
    plt.close()

    # 3. Miner Activity (using original string names)
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df['miner_id_original'], order = df['miner_id_original'].value_counts().index)
    plt.title('Block Mining Activity by Miner')
    plt.xlabel('Number of Blocks Mined')
    plt.ylabel('Miner ID')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'miner_activity.png'))
    plt.close()
    
    # 4. Key Features Over Time
    time_features = ['num_transactions', 'total_amount_transacted', 'time_since_last_block']
    plt.figure(figsize=(15, 8))
    for i, feature in enumerate(time_features):
        if feature in df.columns:
            plt.subplot(len(time_features), 1, i + 1)
            plt.plot(df['block_index'], df[feature])
            plt.title(f'{feature} Over Time (Block Index)')
            plt.ylabel(feature)
    plt.xlabel('Block Index')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'key_features_over_time.png'))
    plt.close()

    logging.info(f"Data exploration plots saved to '{vis_dir}' directory.")


def visualize_model_performance(history, train_mae_loss, threshold, model, X_test, feature_columns, vis_dir):
    """Generates and saves plots related to model performance."""
    logging.info("Generating model performance visualizations...")
    
    # 1. Training & Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs'); plt.ylabel('Loss (MAE)'); plt.xlabel('Epoch')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'unsupervised_loss_history.png'))
    plt.close()

    # 2. Reconstruction Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(train_mae_loss, bins=50, kde=True, label='Train Reconstruction Errors')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Anomaly Threshold ({threshold:.4f})')
    plt.title('Reconstruction Error Distribution on Training Data')
    plt.xlabel('Mean Absolute Error (MAE)'); plt.ylabel('Frequency')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'reconstruction_errors_distribution.png'))
    plt.close()
    
    # 3. Feature-wise Reconstruction Error
    X_train_pred = model.predict(X_test)
    feature_mae = np.mean(np.abs(X_train_pred - X_test), axis=(0, 1))
    feature_error_df = pd.DataFrame({'feature': feature_columns, 'mae': feature_mae}).sort_values('mae', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='mae', y='feature', data=feature_error_df)
    plt.title('Mean Absolute Error by Feature')
    plt.xlabel('Mean Absolute Error (MAE)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_wise_reconstruction_error.png'))
    plt.close()

    # 4. Plot Example Reconstructions
    def plot_example(original, reconstructed, error, title, path):
        plt.figure(figsize=(15, 10))
        for i in range(original.shape[1]):
            plt.subplot(original.shape[1], 1, i + 1)
            plt.plot(original[:, i], 'b', label='Original')
            plt.plot(reconstructed[:, i], 'r', linestyle='--', label='Reconstructed')
            plt.ylabel(feature_columns[i], rotation=0, labelpad=40)
            plt.xticks([])
        plt.suptitle(f'{title} (Overall MAE: {error:.4f})', fontsize=16)
        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        plt.close()

    test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(test_pred - X_test), axis=(1, 2))
    
    # Low-error (normal) example
    normal_idx = np.argmin(test_mae_loss)
    plot_example(X_test[normal_idx], test_pred[normal_idx], test_mae_loss[normal_idx],
                 'Normal Sequence Reconstruction Example', os.path.join(vis_dir, 'reconstruction_example_normal.png'))
                 
    # High-error (anomalous) example
    anomaly_idx = np.argmax(test_mae_loss)
    plot_example(X_test[anomaly_idx], test_pred[anomaly_idx], test_mae_loss[anomaly_idx],
                 'Anomalous Sequence Reconstruction Example', os.path.join(vis_dir, 'reconstruction_example_anomaly.png'))

    logging.info(f"Model performance plots saved to '{vis_dir}' directory.")


# === Main Logic ===

def create_sequences(data, sequence_length):
    """Creates overlapping sequences from the data."""
    xs = []
    if len(data) < sequence_length: return np.array(xs)
    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length)])
    return np.array(xs)

def main():
    # --- 1. Setup Environment ---
    for d in [DATA_DIR, MODEL_DIR, VIS_DIR]: os.makedirs(d, exist_ok=True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU detected and memory growth set for {len(gpus)} device(s).")
        except RuntimeError as e: logging.error(e)
    else: logging.warning("No GPU detected. TensorFlow will run on CPU.")

    # --- 2. Load and Preprocess Data ---
    df = pd.read_csv(DATA_PATH)
    
    # Store original miner IDs for visualization before encoding them
    df['miner_id_original'] = df['miner_id']
    
    # Create and save the mapping for live detection
    miner_id_mapping = {cat: i for i, cat in enumerate(df['miner_id'].astype('category').cat.categories)}
    miner_id_mapping['unknown'] = -1 # Handle new miners not seen in training
    
    # Convert miner_id to numerical codes for training
    df['miner_id'] = df['miner_id'].astype('category').cat.codes
    
    features_to_use = [col for col in df.columns if col not in ['block_index', 'miner_id_original']]
    features_df = df[features_to_use]
    
    # --- 3. Data Exploration Visualization ---
    visualize_data_exploration(df, features_to_use, VIS_DIR)
    
    # --- 4. Scale and Create Sequences ---
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)
    X = create_sequences(scaled_features, SEQUENCE_LENGTH)
    
    if X.shape[0] == 0:
        logging.error("Not enough data to create sequences. Please generate more data.")
        return
        
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # --- 5. Build LSTM Autoencoder Model ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)
    encoder = LSTM(64, activation='relu', return_sequences=False)(inputs)
    encoder = RepeatVector(input_shape[0])(encoder)
    decoder = LSTM(64, activation='relu', return_sequences=True)(encoder)
    
    # ************************ FIX IS HERE ************************
    # The number of features is at index 1 of the input_shape tuple.
    output = TimeDistributed(Dense(input_shape[1]))(decoder)
    # *************************************************************
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    
    # --- 6. Train the Model ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    
    history = model.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping, reduce_lr], verbose=1)
    
    # --- 7. Determine Anomaly Threshold ---
    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=(1, 2))
    threshold = np.mean(train_mae_loss) + THRESHOLD_MULTIPLIER * np.std(train_mae_loss)
    logging.info(f"Calculated anomaly threshold: {threshold:.4f}")
    
    # --- 8. Model Performance Visualization ---
    visualize_model_performance(history, train_mae_loss, threshold, model, X_test, features_to_use, VIS_DIR)
    
    # --- 9. Save All Model Assets ---
    model.save(os.path.join(MODEL_DIR, 'lstm_autoencoder_model.keras'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'data_scaler.joblib'))
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'w') as f: json.dump(features_to_use, f)
    with open(os.path.join(MODEL_DIR, 'miner_id_mapping.json'), 'w') as f: json.dump(miner_id_mapping, f, indent=4)
    with open(os.path.join(MODEL_DIR, 'anomaly_threshold.json'), 'w') as f: json.dump({'threshold': threshold}, f)
    logging.info("All model assets saved successfully.")

if __name__ == "__main__":
    main()