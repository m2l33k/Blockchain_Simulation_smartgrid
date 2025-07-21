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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.saving import register_keras_serializable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sequences(data, labels, sequence_length=20):
    X, y = [], []
    if len(data) <= sequence_length:
        return np.array(X), np.array(y) 
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(labels[i + sequence_length])
    return np.array(X), np.array(y)

def plot_training_history(history, file_path='training_history.png'):
    plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train Acc'); plt.plot(history.history['val_accuracy'], label='Val Acc'); plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(file_path); plt.close()
    logging.info(f"Training history plot saved to {file_path}")

def plot_confusion_matrix(y_true, y_pred, file_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred); disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']); disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix'); plt.savefig(file_path); plt.close()
    logging.info(f"Confusion matrix saved to {file_path}")

@register_keras_serializable()
class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

def main():
    DATA_DIR, MODEL_DIR, VIS_DIR = 'data', 'saved_models', 'visualized_data'
    for d in [DATA_DIR, MODEL_DIR, VIS_DIR]: os.makedirs(d, exist_ok=True)

    data_path = os.path.join(DATA_DIR, 'featurized_labeled_data.csv')
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at '{data_path}'. Run the data generation script first.")
        return
    df = pd.read_csv(data_path)
    
    SEQUENCE_LENGTH = 20
    
    MINIMUM_DATA_ROWS = SEQUENCE_LENGTH + 50 
    if df.shape[0] < MINIMUM_DATA_ROWS:
        logging.error(f"Not enough data to train. Found {df.shape[0]} rows, but require at least {MINIMUM_DATA_ROWS}.")
        return
        
    features_df = df.drop(columns=['block_index', 'is_anomaly', 'anomaly_type'])
    labels_series = df['is_anomaly']

    if labels_series.nunique() < 2:
        logging.warning("DATASET WARNING: The dataset contains only one class. Model will not learn effectively.")
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    X, y = create_sequences(scaled_features, labels_series.values, SEQUENCE_LENGTH)
    
    if len(X) == 0:
        logging.error("Failed to create any sequences from the data. The dataset is too small.")
        return

    stratify_option = y if len(np.unique(y)) > 1 else None
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )
    
    stratify_train_val = y_train_val if len(np.unique(y_train_val)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=stratify_train_val
    )

    logging.info(f"Data split into: {len(X_train)} training samples, {len(X_val)} validation samples, and {len(X_test)} test samples.")

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    logging.info(f"Calculated class weights to handle imbalance: {class_weight_dict}")
    
    logging.info("Building the Hybrid LSTM-Transformer model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)
    
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = TransformerEncoderBlock(embed_dim=128, num_heads=8, ff_dim=128, rate=0.3)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    logging.info("Starting model training...")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,
        validation_data=(X_val, y_val), 
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr], 
        verbose=1
    )

    logging.info("Evaluating final model on the untouched test set...")
    y_pred_proba = model.predict(X_test)
    y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()

    logging.info("--- Final Classification Report ---")
    print(classification_report(y_test, y_pred_classes, target_names=['Normal', 'Anomaly']))
    
    plot_training_history(history, file_path=os.path.join(VIS_DIR, 'history_hybrid.png'))
    plot_confusion_matrix(y_test, y_pred_classes, file_path=os.path.join(VIS_DIR, 'confusion_matrix_hybrid.png'))

    model_path = os.path.join(MODEL_DIR, 'anomaly_detection_hybrid_model.keras')
    model.save(model_path)
    logging.info(f"Hybrid model saved to {model_path}")
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'data_scaler.joblib'))
    logging.info(f"Data scaler saved to {os.path.join(MODEL_DIR, 'data_scaler.joblib')}")
    
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(features_df.columns.tolist(), f)
    logging.info(f"Feature columns saved to {os.path.join(MODEL_DIR, 'feature_columns.json')}")

if __name__ == "__main__":
    main()