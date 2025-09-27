
import pandas as pd
import numpy as np
import logging
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow and Keras for the Deep Learning Model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = 'data'
MODEL_DIR = 'saved_models'
VIS_DIR = 'visualized_data'
DATA_PATH = os.path.join(DATA_DIR, 'featurized_block_data.csv')
SEQUENCE_LENGTH = 10
THRESHOLD_MULTIPLIER = 2.0 
PSEUDO_LABEL_QUANTILE = 0.95 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def visualize_data_exploration(df, features, vis_dir):
    """Generates and saves all plots for initial data exploration."""
    logging.info("Generating data exploration visualizations...")
    
    plt.figure(figsize=(20, 15)); num_features = len(features); cols = 4; rows = (num_features + cols - 1) // cols
    for i, col in enumerate(features):
        plt.subplot(rows, cols, i + 1); sns.histplot(df[col], kde=True, bins=30); plt.title(f'Distribution of {col}', fontsize=10); plt.xlabel(''); plt.ylabel('')
    plt.suptitle('Feature Distributions', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(vis_dir, 'feature_distributions.png')); plt.close()

    plt.figure(figsize=(12, 10)); correlation_matrix = df[features].corr(); sns.heatmap(correlation_matrix, cmap='coolwarm'); plt.title('Feature Correlation Heatmap'); plt.tight_layout(); plt.savefig(os.path.join(vis_dir, 'feature_correlation_heatmap.png')); plt.close()

    plt.figure(figsize=(10, 8)); sns.countplot(y=df['miner_id_original'], order=df['miner_id_original'].value_counts().index); plt.title('Block Mining Activity by Miner'); plt.xlabel('Number of Blocks Mined'); plt.ylabel('Miner ID'); plt.tight_layout(); plt.savefig(os.path.join(vis_dir, 'miner_activity.png')); plt.close()
    
    time_features = ['num_transactions', 'total_amount_transacted', 'time_since_last_block']
    plt.figure(figsize=(15, 8))
    for i, feature in enumerate(time_features):
        if feature in df.columns:
            plt.subplot(len(time_features), 1, i + 1); plt.plot(df['block_index'], df[feature]); plt.title(f'{feature} Over Time (Block Index)'); plt.ylabel(feature)
    plt.xlabel('Block Index'); plt.tight_layout(); plt.savefig(os.path.join(vis_dir, 'key_features_over_time.png')); plt.close()
    logging.info(f"Data exploration plots saved to '{vis_dir}'.")

def visualize_model_performance(history, train_mae_loss, raw_threshold, model, X_test, feature_columns, vis_dir):
    """Generates and saves all plots related to the LSTM autoencoder's specific performance."""
    logging.info("Generating LSTM autoencoder performance visualizations...")
    
    plt.figure(figsize=(8, 5)); plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss'); plt.title('LSTM Model Loss Over Epochs'); plt.ylabel('Loss (MAE)'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(vis_dir, 'lstm_loss_history.png')); plt.close()

    plt.figure(figsize=(10, 6)); sns.histplot(train_mae_loss, bins=50, kde=True, label='Train Reconstruction Errors'); plt.axvline(raw_threshold, color='r', linestyle='--', label=f'Anomaly Threshold ({raw_threshold:.4f})'); plt.title('LSTM Reconstruction Error Distribution on Training Data'); plt.xlabel('Mean Absolute Error (MAE)'); plt.ylabel('Frequency'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(vis_dir, 'lstm_reconstruction_errors_distribution.png')); plt.close()
    
    X_pred = model.predict(X_test); feature_mae = np.mean(np.abs(X_pred - X_test), axis=(0, 1)); feature_error_df = pd.DataFrame({'feature': feature_columns, 'mae': feature_mae}).sort_values('mae', ascending=False)
    plt.figure(figsize=(12, 8)); sns.barplot(x='mae', y='feature', data=feature_error_df); plt.title('Mean Absolute Error by Feature'); plt.xlabel('Mean Absolute Error (MAE)'); plt.ylabel('Feature'); plt.tight_layout(); plt.savefig(os.path.join(vis_dir, 'lstm_feature_wise_reconstruction_error.png')); plt.close()

    def plot_example(original, reconstructed, error, title, path):
        plt.figure(figsize=(15, 10))
        for i in range(original.shape[1]):
            plt.subplot(original.shape[1], 1, i + 1); plt.plot(original[:, i], 'b', label='Original'); plt.plot(reconstructed[:, i], 'r', linestyle='--', label='Reconstructed'); plt.ylabel(feature_columns[i], rotation=0, labelpad=40); plt.xticks([])
        if i == 0: plt.legend()
        plt.suptitle(f'{title} (Overall MAE: {error:.4f})', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(path); plt.close()

    test_pred = model.predict(X_test); test_mae_loss = np.mean(np.abs(test_pred - X_test), axis=(1, 2))
    normal_idx = np.argmin(test_mae_loss); plot_example(X_test[normal_idx], test_pred[normal_idx], test_mae_loss[normal_idx], 'Normal Sequence Reconstruction Example', os.path.join(vis_dir, 'reconstruction_example_normal.png'))
    anomaly_idx = np.argmax(test_mae_loss); plot_example(X_test[anomaly_idx], test_pred[anomaly_idx], test_mae_loss[anomaly_idx], 'Anomalous Sequence Reconstruction Example', os.path.join(vis_dir, 'reconstruction_example_anomaly.png'))
    logging.info(f"LSTM performance plots saved to '{vis_dir}'.")

def visualize_ensemble_performance(y_true_pseudo, scores_dict, thresholds_dict, vis_dir):
    """Generates all plots related to the comparative performance of the ensemble."""
    logging.info("Generating ensemble performance visualizations...")
    
    plt.figure(figsize=(12, 7))
    for model_name, scores in scores_dict.items():
        sns.histplot(scores, bins=50, kde=True, label=f'{model_name} Scores', alpha=0.6)
    for model_name, threshold in thresholds_dict.items():
        plt.axvline(threshold, linestyle='--', label=f'{model_name} Threshold ({threshold:.4f})')
    plt.title('Comparison of Anomaly Score Distributions'); plt.xlabel('Normalized Score / Probability'); plt.ylabel('Frequency'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(vis_dir, 'score_distributions_comparison.png')); plt.close()
    
    final_scores = scores_dict['Stacked Meta-Learner']
    final_threshold = thresholds_dict['Stacked Meta-Learner']
    plt.figure(figsize=(10, 6)); sns.histplot(final_scores, bins=50, kde=True); plt.axvline(final_threshold, color='r', linestyle='--', label=f'Threshold ({final_threshold:.4f})'); plt.title('Final Anomaly Score Distribution (from Meta-Learner)'); plt.xlabel('Anomaly Probability'); plt.ylabel('Frequency'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(vis_dir, 'final_scores_distribution.png')); plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for model_name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true_pseudo, scores); roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:0.2f})')
        precision, recall, _ = precision_recall_curve(y_true_pseudo, scores)
        ax2.plot(recall, precision, lw=2, label=f'{model_name}')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); ax1.set_xlim([0.0, 1.0]); ax1.set_ylim([0.0, 1.05]); ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate'); ax1.set_title('Receiver Operating Characteristic (ROC) Curve'); ax1.legend(loc="lower right"); ax1.grid(True)
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Precision-Recall Curve'); ax2.legend(loc="lower left"); ax2.grid(True)
    plt.suptitle('Model Performance Evaluation on Pseudo-Labels'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(vis_dir, 'classification_curves.png')); plt.close()

    num_models = len(thresholds_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), squeeze=False)
    for i, (model_name, threshold) in enumerate(thresholds_dict.items()):
        ax = axes[0, i]
        scores = scores_dict[model_name]
        y_pred_pseudo = (scores > threshold).astype(int); cm = confusion_matrix(y_true_pseudo, y_pred_pseudo)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly']); disp.plot(ax=ax, cmap=plt.cm.Blues); ax.set_title(f'CM: {model_name}')
    plt.suptitle('Confusion Matrices on Pseudo-Labels'); plt.tight_layout(rect=[0, 0.03, 1, 0.90]); plt.savefig(os.path.join(vis_dir, 'confusion_matrices.png')); plt.close()
    logging.info(f"Ensemble performance plots saved to '{vis_dir}'.")

def create_sequences(data, sequence_length):
    """Creates overlapping sequences from the data."""
    xs = []
    if len(data) < sequence_length: return np.array(xs)
    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length)])
    return np.array(xs)

def main():
    for d in [DATA_DIR, MODEL_DIR, VIS_DIR]: os.makedirs(d, exist_ok=True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU detected and memory growth set for {len(gpus)} device(s).")
        except RuntimeError as e: logging.error(e)
    else: logging.warning("No GPU detected. TensorFlow will run on CPU.")
    

    logging.info("Loading and preprocessing data...")
    df = pd.read_csv(DATA_PATH)
    df['miner_id_original'] = df['miner_id']
    miner_id_mapping = {cat: i for i, cat in enumerate(df['miner_id'].astype('category').cat.categories)}; miner_id_mapping['unknown'] = -1
    df['miner_id'] = df['miner_id'].astype('category').cat.codes
    features_to_use = [col for col in df.columns if col not in ['block_index', 'miner_id_original']]
    

    visualize_data_exploration(df, features_to_use, VIS_DIR)
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features_to_use])
    X = create_sequences(scaled_features, SEQUENCE_LENGTH)
    if X.shape[0] == 0: logging.error("Not enough data to create sequences."); return
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    logging.info("--- Training Stacked LSTM Autoencoder Model (3 layers) ---")
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)
    encoder = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(inputs)
    encoder = Bidirectional(LSTM(64, activation='relu', return_sequences=False))(encoder)
    encoder = RepeatVector(input_shape[0])(encoder)
    decoder = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(encoder)
    decoder = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(decoder)
    output = TimeDistributed(Dense(input_shape[1]))(decoder)
    lstm_model = Model(inputs=inputs, outputs=output)
    lstm_model.compile(optimizer='adam', loss='mae')
    lstm_model.summary()
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)]
    history = lstm_model.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=1)
    
    logging.info("--- Training Isolation Forest Model ---")
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    if_model = IsolationForest(n_estimators=150, contamination='auto', random_state=42, n_jobs=-1)
    if_model.fit(X_train_flat)

    logging.info("--- Generating and Normalizing Base Model Scores ---")
    X_train_pred_lstm = lstm_model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred_lstm - X_train), axis=(1, 2))
    train_if_scores = -1 * if_model.decision_function(X_train_flat)
    lstm_score_scaler = MinMaxScaler(); if_score_scaler = MinMaxScaler()
    norm_lstm_scores_train = lstm_score_scaler.fit_transform(train_mae_loss.reshape(-1, 1)).flatten()
    norm_if_scores_train = if_score_scaler.fit_transform(train_if_scores.reshape(-1, 1)).flatten()

    logging.info("--- Training Meta-Learner ---")
    X_meta_train = np.column_stack([norm_lstm_scores_train, norm_if_scores_train])
    combined_scores = (norm_lstm_scores_train + norm_if_scores_train) / 2.0
    pseudo_label_threshold = np.quantile(combined_scores, PSEUDO_LABEL_QUANTILE)
    y_meta_train_pseudo = (combined_scores > pseudo_label_threshold).astype(int)
    logging.info(f"Created {np.sum(y_meta_train_pseudo)} pseudo-anomaly labels for training.")
    meta_learner = LogisticRegression(class_weight='balanced', random_state=42)
    meta_learner.fit(X_meta_train, y_meta_train_pseudo)

    logging.info("--- Calculating Final Thresholds for All Models ---")
    final_anomaly_scores_train = meta_learner.predict_proba(X_meta_train)[:, 1]
    
    thresholds = {
        "lstm_normalized": np.mean(norm_lstm_scores_train) + THRESHOLD_MULTIPLIER * np.std(norm_lstm_scores_train),
        "iforest_normalized": np.mean(norm_if_scores_train) + THRESHOLD_MULTIPLIER * np.std(norm_if_scores_train),
        "stacked_final": np.mean(final_anomaly_scores_train) + THRESHOLD_MULTIPLIER * np.std(final_anomaly_scores_train)
    }
    thresholds["stacked_final"] = min(thresholds["stacked_final"], 1.0)
    logging.info(f"Calculated Thresholds: {thresholds}")

    lstm_raw_threshold = np.mean(train_mae_loss) + THRESHOLD_MULTIPLIER * np.std(train_mae_loss)
    visualize_model_performance(history, train_mae_loss, lstm_raw_threshold, lstm_model, X_test, features_to_use, VIS_DIR)
    
    scores_dict_viz = {
        'LSTM Autoencoder': norm_lstm_scores_train,
        'Isolation Forest': norm_if_scores_train,
        'Stacked Meta-Learner': final_anomaly_scores_train
    }
    thresholds_dict_viz = {
        'LSTM Autoencoder': thresholds["lstm_normalized"],
        'Isolation Forest': thresholds["iforest_normalized"],
        'Stacked Meta-Learner': thresholds["stacked_final"]
    }
    visualize_ensemble_performance(y_meta_train_pseudo, scores_dict_viz, thresholds_dict_viz, VIS_DIR)
    
    logging.info("--- Saving All Model Assets ---")
    lstm_model.save(os.path.join(MODEL_DIR, 'lstm_autoencoder_model.keras'))
    joblib.dump(if_model, os.path.join(MODEL_DIR, 'isolation_forest_model.joblib'))
    joblib.dump(meta_learner, os.path.join(MODEL_DIR, 'meta_learner_model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'data_scaler.joblib'))
    joblib.dump(lstm_score_scaler, os.path.join(MODEL_DIR, 'lstm_score_scaler.joblib'))
    joblib.dump(if_score_scaler, os.path.join(MODEL_DIR, 'if_score_scaler.joblib'))
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'w') as f: json.dump(features_to_use, f)
    with open(os.path.join(MODEL_DIR, 'miner_id_mapping.json'), 'w') as f: json.dump(miner_id_mapping, f, indent=4)
    with open(os.path.join(MODEL_DIR, 'thresholds.json'), 'w') as f: json.dump(thresholds, f, indent=4)
    
    logging.info("====== Training Pipeline Complete. All artifacts saved. ======")

if __name__ == "__main__":
    main()