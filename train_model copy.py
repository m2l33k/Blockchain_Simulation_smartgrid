# train_model.py (Final Corrected Version)

import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import joblib 
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sequences(data, labels, sequence_length=20):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(labels[i + sequence_length])
    return np.array(X), np.array(y)

def plot_training_history(history, file_path='training_history.png'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, file_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Test Set')
    plt.savefig(file_path)
    plt.close()

def main():
    DATA_DIR = 'data'
    MODEL_DIR = 'saved_models'
    VIS_DIR = 'visualized_data'

    for directory in [DATA_DIR, MODEL_DIR, VIS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    logging.info("Loading featurized data...")
    data_path = os.path.join(DATA_DIR, 'featurized_labeled_data.csv')
    if not os.path.exists(data_path):
        logging.error(f"Data file not found. Please run the training pipeline first.")
        return

    df = pd.read_csv(data_path)
    SEQUENCE_LENGTH = 20
    if df.shape[0] < SEQUENCE_LENGTH * 2:
        logging.error("Not enough data to train. Please run a longer simulation.")
        return

    features_df = df.drop(columns=['block_index', 'is_anomaly', 'anomaly_type'])
    labels_series = df['is_anomaly']

    logging.info(f"Scaling features and creating sequences of length {SEQUENCE_LENGTH}...")
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)
    X, y = create_sequences(scaled_features, labels_series.values, SEQUENCE_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    logging.info(f"Using class weights: {class_weight_dict}")

    logging.info("Building the Enhanced LSTM model...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True), Dropout(0.3),
        LSTM(64, return_sequences=True), Dropout(0.3),
        LSTM(32, return_sequences=False), Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    logging.info("Starting model training...")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, epochs=100, batch_size=64, validation_split=0.2,
        class_weight=class_weight_dict, callbacks=[early_stopping, reduce_lr], verbose=1
    )

    logging.info("Evaluating final model on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_proba = model.predict(X_test)
    y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()

    logging.info(f"Final Test Loss: {test_loss:.4f}")
    logging.info(f"Final Test Accuracy: {test_accuracy:.4f}")
    logging.info("--- Final Classification Report ---")
    print(classification_report(y_test, y_pred_classes, target_names=['Normal', 'Anomaly']))
    
    plot_training_history(history, file_path=os.path.join(VIS_DIR, 'training_history_final.png'))
    plot_confusion_matrix(y_test, y_pred_classes, file_path=os.path.join(VIS_DIR, 'confusion_matrix_final.png'))

    logging.info("Saving all model assets...")
    
    model_path = os.path.join(MODEL_DIR, 'anomaly_detection_model.keras')
    model.save(model_path)
    logging.info(f"Enhanced model saved to {model_path}")
    
    scaler_path = os.path.join(MODEL_DIR, 'data_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logging.info(f"Data scaler saved to {scaler_path}")

    feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.json')
    with open(feature_columns_path, 'w') as f:
        json.dump(features_df.columns.tolist(), f)
    logging.info(f"Feature columns saved to {feature_columns_path}")
    
    val_start_index = int(len(X_train) * 0.8)
    X_val, y_val = X_train[val_start_index:], y_train[val_start_index:]
    X_val_normal = X_val[y_val == 0]
    
    if len(X_val_normal) > 0:
        normal_scores = model.predict(X_val_normal, verbose=0).flatten()
        alert_threshold = np.mean(normal_scores) + (3 * np.std(normal_scores))
        alert_threshold = max(0.6, min(0.98, alert_threshold))
        
        logging.info(f"Calculated adaptive alert threshold: {alert_threshold:.4f}")
        
        threshold_path = os.path.join(MODEL_DIR, 'alert_threshold.json')
        with open(threshold_path, 'w') as f:
            # FIX: Convert the numpy float to a standard Python float
            json.dump({'threshold': float(alert_threshold)}, f)
        logging.info(f"Alert threshold saved to {threshold_path}")
    else:
        logging.warning("Could not calculate adaptive threshold: no normal data in validation set.")

if __name__ == "__main__":
    main()