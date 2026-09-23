from pathlib import Path

import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.inference import (
    display_intent,
    load_classifier_artifacts,
    predict_top3,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    classifier, intent_to_category = (
        load_classifier_artifacts()
    )

    return (
        embedding_model,
        classifier,
        intent_to_category,
    )


st.set_page_config(
    page_title="AI Support Ticket Classifier",
    page_icon="🤖",
    layout="centered",
)

try:
    embedding_model, classifier, intent_to_category = load_models()
except Exception as exc:
    st.error("The model could not be loaded. Please try again later.")
    st.exception(exc)
    st.stop()

st.title("🤖 AI Support Ticket Classifier")
st.write(
    "Classify an English customer-support message by intent and category "
    "using semantic sentence embeddings."
)

with st.expander("How it works"):
    st.write(
        "The app encodes the message with `all-MiniLM-L6-v2` and sends the "
        "384-dimensional embedding to a Logistic Regression classifier trained "
        "on 27 support intents."
    )
    st.caption(
        "This is a closed-set classifier: every message is assigned to one of "
        "the known intents, even when the message is outside the training domain."
    )

example = st.selectbox(
    "Optional example",
    [
        "Write my own message",
        "I forgot my password",
        "My order has not arrived yet",
        "Can you change the address for my order?",
        "I need help",
    ],
)

default_text = "" if example == "Write my own message" else example

with st.form("classifier_form"):
    text = st.text_area(
        "Customer message",
        value=default_text,
        placeholder="Example: I forgot my password",
        height=140,
    )
    submitted = st.form_submit_button("Classify", type="primary")

if submitted:
    if not text.strip():
        st.warning("Please enter a customer message.")
    else:
        results = predict_top3(
            text.strip(),
            embedding_model,
            classifier,
            intent_to_category,
        )
        primary = results[0]
        confidence = primary["confidence"]

        st.subheader("Prediction")

        col1, col2 = st.columns(2)
        col1.metric("Category", primary["category"])
        col2.metric("Probability", f"{confidence:.1f}%")

        st.markdown(
            f"**Intent:** {display_intent(primary['intent'])} "
            f"(`{primary['intent']}`)"
        )

        if confidence >= 70:
            st.success("High-confidence prediction.")
        elif confidence >= 40:
            st.warning("Medium-confidence prediction — the message may be ambiguous.")
        else:
            st.error("Low-confidence prediction — human review is recommended.")

        st.subheader("Top 3 predictions")
        for rank, result in enumerate(results, start=1):
            st.write(
                f"**{rank}. {display_intent(result['intent'])}** "
                f"(`{result['intent']}`) — {result['category']} — "
                f"{result['confidence']:.1f}%"
            )
            st.progress(min(result["confidence"] / 100, 1.0))

st.divider()
st.caption(
    "Final model: all-MiniLM-L6-v2 sentence embeddings + Logistic Regression. "
    "Probability estimates should not be interpreted as certainty."
)
