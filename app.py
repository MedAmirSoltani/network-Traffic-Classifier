import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import StringIO
from PIL import Image

# Load model and preprocessing object
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("mlp_model.pkl", "rb") as f:
    mlp_model = pickle.load(f)

# Feature names for the CSV file
original_feature_names = [
    'BRST_COUNT', 'REV_MORE', 'PKT_LENGTHS_STD', 'INTERVALS_MAX',
    'INTERVALS_75', 'BRST_BYTES_STD', 'BRST_INTERVALS_STD',
    'BRST_INTERVALS_50', 'BRST_DURATION_STD', 'PCA_DBI_BRST_TIME_START_0',
    'PCA_DBI_BRST_TIME_START_1', 'PCA_PKT_LENGTHS_1',
    'PCA_BRST_INTERVALS_0', 'PCA_BRST_INTERVALS_1', 'PCA_BRST_DURATION_0',
    'PCA_BRST_DURATION_1', 'BRST_BYTES_0', 'BRST_BYTES_2', 'BRST_BYTES_3',
    'BRST_BYTES_4', 'BRST_BYTES_5', 'BRST_BYTES_6', 'BRST_BYTES_7',
    'BRST_BYTES_8', 'BRST_BYTES_9', 'PPI_PKT_DIRECTIONS_mean',
    'PPI_PKT_DIRECTIONS_std', 'PKT_TIMES_mean'
]

# User-friendly feature names for manual input
feature_names = [
    'Burst Count', 'Reverse Traffic', 'Packet Length Variation',
    'Max Time Between Packets', '75th Percentile Time Between Packets', 'Burst Byte Variation',
    'Burst Time Variation', 'Median Time Between Bursts', 'Burst Duration Variation',
    'Burst Start Time (1)', 'Burst Start Time (2)', 'Packet Length (Analysis)',
    'Burst Time Variation (1)', 'Burst Time Variation (2)', 'Burst Duration (Analysis 1)',
    'Burst Duration (Analysis 2)', 'Burst Bytes (0-10)', 'Burst Bytes (10-20)', 'Burst Bytes (20-30)',
    'Burst Bytes (30-40)', 'Burst Bytes (40-50)', 'Burst Bytes (50-60)', 'Burst Bytes (60-70)',
    'Burst Bytes (70-80)', 'Burst Bytes (80-90)', 'Avg Packet Direction',
    'Packet Direction Variation', 'Avg Packet Timing'
]

# Streamlit UI Design
st.set_page_config(page_title="Traffic Classifier", page_icon="🛡️", layout="wide")

# Sidebar for navigation
st.sidebar.image("hacker.png", width=350)
st.sidebar.title("🔒 Traffic Classifier")
st.sidebar.write("Classify HTTPS traffic as **Heavy** or **Not Heavy** based on packet statistics.")

# Input selection
option = st.sidebar.radio("Choose Input Method", ["📂 Upload CSV", "✍️ Manual Entry"])

# Function for prediction
def predict(data):
    """Applies preprocessing and makes predictions."""
    scaled_input = scaler.transform(data)
    pca_input = pca.transform(scaled_input)
    prediction = mlp_model.predict(pca_input)
    return ["Heavy Traffic" if pred == 1 else "Not Heavy" for pred in prediction]

# Function to highlight heavy traffic in red
def highlight_rows(row):
    return ["background-color: #FF5252; color: white; font-weight: bold;" if row["Prediction"] == "Heavy Traffic" else "" for _ in row]

# CSV Upload
if option == "📂 Upload CSV":
    st.header("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with the correct format", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Check if all required columns exist
        if set(original_feature_names).issubset(df.columns):
            predictions = predict(df[original_feature_names])
            df["Prediction"] = predictions
            st.success("✅ Prediction complete!")
            
            # Apply styling
            styled_df = df.style.apply(highlight_rows, axis=1)
            st.dataframe(styled_df)
            
            st.download_button(label="⬇️ Download Results", data=df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
        else:
            st.error("❌ The uploaded file does not have the correct feature columns.")

# Manual Entry
elif option == "✍️ Manual Entry":
    st.header("✍️ Enter Feature Values Manually")
    
    input_values = []
    col1, col2 = st.columns(2)

    # Splitting the feature inputs into two columns for better layout
    for i, feature in enumerate(feature_names):
        with col1 if i % 2 == 0 else col2:
            value = st.number_input(f"{feature}", value=0.0)
            input_values.append(value)

    # Predict button
    if st.button("🔍 Predict"):
        input_array = np.array(input_values).reshape(1, -1)
        result = predict(input_array)[0]
        
        # Highlight if heavy traffic
        if result == "Heavy Traffic":
            st.error(f"### 🛑 Prediction: **{result}**", icon="⚠️")
        else:
            st.success(f"### ✅ Prediction: **{result}**")

# Footer
st.sidebar.write("---")
st.sidebar.write("🛠️ Developed for cybersecurity analytics.")
st.sidebar.write("💡 Ensure correct data format for accurate predictions.")
