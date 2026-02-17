import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="AI Spam Detection System",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()


# Example: load or define confusion matrix
@st.cache_resource
def load_confusion_matrix():
    # Replace this with your actual cm from test set
    # cm = confusion_matrix(y_test, y_pred)
    cm = [[950, 15], [20, 915]]  # TN, FP, FN, TP (example)
    return cm


cm = load_confusion_matrix()


# Header
st.title("📩 AI Spam Detection System")
st.markdown(
    """
    **Detect whether a message is Spam or Not Spam using Machine Learning.**
    
    Enter any SMS or email text below and click **Predict** to see the result.
    """
)

# Input section
st.markdown("### 📝 Enter your message")
input_sms = st.text_area(
    "Message text:",
    height=150,
    placeholder="Type or paste the SMS/email you want to check here..."
)

# Prediction section
if st.button("🔍 Predict", type="primary"):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        with st.spinner("Analyzing message..."):
            transformed_sms = vectorizer.transform([input_sms])
            prediction = model.predict(transformed_sms)[0]

            try:
                probability = model.predict_proba(transformed_sms)[0]
                confidence = max(probability) * 100
            except AttributeError:
                confidence = None

        if prediction == 1:
            if confidence is not None:
                st.error(
                    f"🚨 **Spam Message** ({confidence:.1f}% confidence)",
                    icon="🚨"
                )
            else:
                st.error("🚨 **Spam Message**", icon="🚨")
        else:
            if confidence is not None:
                st.success(
                    f"✅ **Not Spam** ({confidence:.1f}% confidence)",
                    icon="✅"
                )
            else:
                st.success("✅ **Not Spam**", icon="✅")


# Sidebar with model info and metrics
st.sidebar.header("📊 Model Information")

st.sidebar.markdown(
    """
    **Tech Stack:**
    - Python
    - Scikit‑learn
    - TF‑IDF Vectorization
    - NLTK (for preprocessing)
    - Streamlit (UI)
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Performance (Test Set)")

# Example metrics (replace with your actual values)
st.sidebar.metric(label="Overall Accuracy", value="98.2%")

st.sidebar.markdown(
    """
    | Class | Precision | Recall | F1‑Score |
    |-------|-----------|--------|----------|
    | Spam  | 0.975     | 0.968  | 0.971    |
    | Ham   | 0.987     | 0.990  | 0.988    |
    """
)

# Confusion matrix in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Confusion Matrix")

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Ham", "Predicted Spam"],
    yticklabels=["Actual Ham", "Actual Spam"],
    ax=ax,
    cbar=False,
)
ax.set_title("Confusion Matrix", fontsize=10, pad=10)
fig.tight_layout()

st.sidebar.pyplot(fig, use_container_width=True)
