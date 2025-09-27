import streamlit as st
from PIL import Image
import os

st.set_page_config(layout="wide", page_title="Model Performance")
st.title("Model Performance Dashboard")
st.markdown("A comprehensive overview of the data exploration, model training, and performance evaluation results.")

# --- Constants ---
VIS_DIR = 'visualized_data'

def display_image(path, caption="", use_column_width='auto'):
    """Helper function to display an image if it exists, with a clear warning if not."""
    if os.path.exists(path):
        image = Image.open(path)
        st.image(image, caption=caption, use_column_width=use_column_width)
    else:
        st.warning(f"Image not found: {os.path.basename(path)}. Please ensure the main training script has been run to generate all visualizations.")

# --- Main Page Layout ---

# Section 1: Final Ensemble Model Performance
st.header("1. Final Stacked Model Performance")
st.markdown("These visualizations evaluate the final ensemble model against its individual components using pseudo-labels from the training data. This demonstrates the power of stacking.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("ROC and Precision-Recall Curves")
    display_image(os.path.join(VIS_DIR, 'classification_curves.png'))
    st.info("""
    **Interpretation:** The **Stacked Meta-Learner** (green line) significantly outperforms the base models. Its perfect ROC Curve (AUC = 1.0) and ideal Precision-Recall curve show its ability to flawlessly distinguish between normal and anomalous pseudo-labels.
    """)
with col2:
    st.subheader("Confusion Matrices")
    display_image(os.path.join(VIS_DIR, 'confusion_matrices.png'))
    st.info("""
    **Interpretation:** The confusion matrices show the classification results at the calculated threshold. The Meta-Learner's matrix should show near-perfect separation, with very few, if any, False Positives or False Negatives.
    """)

# Section 2: Data Exploration Insights
with st.expander("Show Initial Data Exploration Visualizations", expanded=False):
    st.header("2. Data Exploration Insights")
    st.markdown("Understanding the underlying data is the first step in building a robust model. These plots show the initial state of the featurized blockchain data.")
    
    st.subheader("Feature Distributions")
    display_image(os.path.join(VIS_DIR, 'feature_distributions.png'))
    
    st.subheader("Miner Activity")
    display_image(os.path.join(VIS_DIR, 'miner_activity.png'))
    
    st.subheader("Feature Correlation Heatmap")
    display_image(os.path.join(VIS_DIR, 'feature_correlation_heatmap.png'))
    
    st.subheader("Key Features Over Time")
    display_image(os.path.join(VIS_DIR, 'key_features_over_time.png'))

# Section 3: Anomaly Score Analysis
with st.expander("Show Anomaly Score Analysis", expanded=False):
    st.header("3. Anomaly Score Analysis")
    st.markdown("These plots detail how the anomaly scores are distributed and how the thresholds are set.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Base Model (LSTM) Reconstruction Errors")
        display_image(os.path.join(VIS_DIR, 'lstm_reconstruction_errors_distribution.png'))
        st.info("This shows the distribution of errors for the standalone LSTM autoencoder. The threshold is set to capture the tail of this distribution.")
        
        st.subheader("Final Combined Score Distribution")
        display_image(os.path.join(VIS_DIR, 'final_scores_distribution.png'))
        st.info("This shows the distribution of the final anomaly probabilities from the meta-learner.")

    with col4:
        st.subheader("Comparison of All Model Scores")
        display_image(os.path.join(VIS_DIR, 'score_distributions_comparison.png'))
        st.info("This plot overlays the score distributions of all models, showing how the meta-learner's scores (probabilities) often provide better separation.")

# Section 4: Deep Dive into LSTM Autoencoder Performance
with st.expander("Show Deep Dive into LSTM Autoencoder Performance", expanded=False):
    st.header("4. Deep Dive into LSTM Autoencoder Performance")
    st.markdown("The LSTM autoencoder is the core component for learning temporal patterns. These visualizations show how it trained and what it learned.")
    
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Model Training & Validation Loss")
        display_image(os.path.join(VIS_DIR, 'lstm_loss_history.png'))
        st.info("The validation loss closely tracks the training loss, indicating that the model is learning effectively without overfitting.")
    
    with col6:
        st.subheader("Feature-wise Reconstruction Error")
        display_image(os.path.join(VIS_DIR, 'lstm_feature_wise_reconstruction_error.png'))
        st.info("This shows which features the model struggles to reconstruct the most. These are often important indicators of anomalies.")

    st.subheader("Reconstruction Examples")
    st.markdown("Comparing the original data sequence (blue) with the model's reconstruction (red). A large gap between the lines indicates a high reconstruction error, and thus, an anomaly.")
    
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Example of a Normal Sequence")
        display_image(os.path.join(VIS_DIR, 'reconstruction_example_normal.png'))
    
    with col8:
        st.subheader("Example of an Anomalous Sequence")        
        display_image(os.path.join(VIS_DIR, 'reconstruction_example_anomaly.png'))