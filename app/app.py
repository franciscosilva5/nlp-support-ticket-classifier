from pathlib import Path

import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def carregar_modelos():
    modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    classificador = joblib.load(
        MODELS_DIR / "semantic_intent_classifier.joblib"
    )

    intent_para_categoria = joblib.load(
        MODELS_DIR / "intent_to_category.joblib"
    )

    return modelo_embeddings, classificador, intent_para_categoria


modelo_embeddings, classificador, intent_para_categoria = carregar_modelos()


def prever(texto):
    embedding = modelo_embeddings.encode([texto])

    probabilidades = classificador.predict_proba(embedding)[0]
    classes = classificador.classes_

    indices_top3 = probabilidades.argsort()[-3:][::-1]

    resultados = []

    for indice in indices_top3:
        intent = classes[indice]

        resultados.append({
            "intent": intent,
            "category": intent_para_categoria[intent],
            "confidence": float(probabilidades[indice]) * 100
        })

    return resultados


st.set_page_config(
    page_title="AI Support Ticket Classifier",
    page_icon="🤖",
    layout="centered"
)

st.title("AI Support Ticket Classifier")

st.write(
    "Classify a customer support message using semantic embeddings "
    "and machine learning."
)

texto = st.text_area(
    "Customer message",
    placeholder="Example: My order has not arrived yet",
    height=140
)

if st.button("Classify", type="primary"):
    if not texto.strip():
        st.warning("Please enter a customer message.")

    else:
        resultados = prever(texto)

        principal = resultados[0]
        confianca = principal["confidence"]

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Category",
                principal["category"]
            )

        with col2:
            st.metric(
                "Intent",
                principal["intent"]
            )

        st.metric(
            "Model confidence",
            f"{confianca:.1f}%"
        )

        if confianca >= 70:
            st.success("High-confidence prediction.")

        elif confianca >= 40:
            st.warning(
                "Medium-confidence prediction. "
                "The message may be ambiguous."
            )

        else:
            st.error(
                "Low-confidence prediction. "
                "Human review is recommended."
            )

        st.subheader("Top 3 predictions")

        for i, resultado in enumerate(resultados, start=1):
            st.write(
                f"**{i}. {resultado['intent']}** "
                f"— {resultado['category']} "
                f"— {resultado['confidence']:.1f}%"
            )

            st.progress(
                min(resultado["confidence"] / 100, 1.0)
            )

st.divider()

st.caption(
    "Model: all-MiniLM-L6-v2 sentence embeddings + "
    "Logistic Regression. Confidence values are model probability "
    "estimates and should not be interpreted as certainty."
)
