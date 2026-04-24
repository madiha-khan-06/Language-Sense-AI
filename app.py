import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Language Sense AI",
    page_icon="🌐",
    layout="wide"
)

# ---------------------------------------------------
# Keep Your Existing Background / UI Colors
# ---------------------------------------------------
st.markdown("""
<style>

/* ==================================================
   FULL BACKGROUND
================================================== */
.stApp {
    background: linear-gradient(135deg, #0c060f 0%, #3f2739 45%, #804e69 100%);
    min-height: 100vh;
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.main .block-container {
    background: transparent !important;
}

/* ==================================================
   SIDEBAR
================================================== */
[data-testid="stSidebar"] {
    background: #0c060f;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ==================================================
   TEXT
================================================== */
html, body, [class*="css"], p, label, div, span, h1, h2, h3 {
    color: #ffffff !important;
    font-family: "Segoe UI", sans-serif;
}

/* ==================================================
   TEXT AREA
================================================== */
textarea {
    background: #ffffff !important;
    color: #0c060f !important;
    border: 1px solid #804e69 !important;
    border-radius: 14px !important;
}

textarea::placeholder {
    color: #3f2739 !important;
}

/* ==================================================
   BUTTON
================================================== */
.stButton > button {
    width: 100%;
    background: #0c060f !important;
    color: #ffffff !important;
    border: 1px solid #804e69 !important;
    border-radius: 14px !important;
    padding: 0.75rem 1rem;
    font-weight: 700;
}

.stButton > button:hover {
    background: #3f2739 !important;
}

/* ==================================================
   FILE UPLOADER
================================================== */
[data-testid="stFileUploader"] {
    background: #0c060f !important;
    border: 1px dashed #804e69 !important;
    border-radius: 14px !important;
    padding: 14px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0c060f !important;
    border: 1px dashed #804e69 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #ffffff !important;
}

[data-testid="stFileUploader"] button {
    background: #3f2739 !important;
    color: #ffffff !important;
    border: 1px solid #804e69 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #804e69 !important;
}

/* ==================================================
   METRIC CARDS
================================================== */
[data-testid="metric-container"] {
    background: rgba(12,6,15,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Upload your own CSV dataset")
st.sidebar.caption("CSV must contain: Text, language")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------------------------------------------
# Load Uploaded Dataset Only
# ---------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# If no file uploaded
if uploaded_file is None:
    st.title("🌐 Language Sense AI")
    st.write("Detect the language of any text using Machine Learning")
    st.info("📁 Please upload a CSV file from the sidebar to continue.")
    st.stop()

df = load_data(uploaded_file)

# ---------------------------------------------------
# Validate Columns
# ---------------------------------------------------
if "Text" not in df.columns or "language" not in df.columns:
    st.error("CSV must contain columns: Text and language")
    st.stop()

# ---------------------------------------------------
# Prepare Data
# ---------------------------------------------------
x = np.array(df["Text"])
y = np.array(df["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.60, random_state=99
)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ---------------------------------------------------
# Main UI
# ---------------------------------------------------
st.title("🌐 Language Sense AI")
st.write("Detect the language of any text using Machine Learning")

# Dataset Info
st.subheader("🗃️ Dataset Info")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows, Columns", f"{df.shape}")

with col2:
    st.metric("Languages", df["language"].nunique())

# Input Text
st.subheader("⌨️ Enter Text")

user_input = st.text_area("Type here...")

# Prediction
if st.button("🔍 Predict Language"):
    if user_input.strip() != "":
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)

        st.success(f"🌐 Predicted Language: {output[0]}")
    else:
        st.warning("Please enter some text!")

# Performance
st.subheader("🎯 Model Performance")
st.metric("Accuracy", f"{accuracy:.2f}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Madiha")