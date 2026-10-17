import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="📊",
    layout="wide"
)

# Load the saved model
@st.cache_resource
def load_model():
    model_path = "best_model.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload best_model.pkl.")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# Application heading
st.title("Credit Risk Analysis System")
st.write(
    "An interactive machine learning demonstration "
    "using a trained Random Forest classifier."
)

st.info(
    "Enter the model's 40 expected input features below. "
    "Some features represent encoded categories rather than "
    "direct customer information."
)

# Expected feature names
features = list(model.feature_names_in_)

st.subheader("Model Input Features")
st.caption(
    "Enter values using the same encoding used when the model was trained."
)

# Collect input values
input_values = {}

with st.form("prediction_form"):

    st.write("### Enter Feature Values")

    columns = st.columns(3)

    for i, feature in enumerate(features):
        with columns[i % 3]:
            input_values[feature] = st.number_input(
                label=feature,
                value=0.0,
                format="%.4f",
                key=feature
            )

    submitted = st.form_submit_button(
        "Predict Credit Risk",
        use_container_width=True
    )

# Generate prediction
if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=features
    )

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Class",
                str(prediction)
            )

        with col2:
            st.metric(
                "Model",
                "Random Forest"
            )

        st.write("### Class Probabilities")

        probability_df = pd.DataFrame({
            "Class": [str(c) for c in classes],
            "Probability": probabilities
        })

        probability_df["Probability (%)"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_df[
                ["Class", "Probability (%)"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            probability_df.set_index("Class")["Probability"]
        )

        st.warning(
            "This is an experimental ML demonstration. "
            "Predictions depend on correctly encoded inputs "
            "and should not be used to make real lending decisions."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.divider()
st.caption(
    "Credit Risk Analysis | Machine Learning Project"
)
