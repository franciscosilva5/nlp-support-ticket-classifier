from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def load_classifier_artifacts():
    classifier = joblib.load(
        MODELS_DIR
        / "semantic_intent_classifier.joblib"
    )

    intent_to_category = joblib.load(
        MODELS_DIR
        / "intent_to_category.joblib"
    )

    return (
        classifier,
        intent_to_category,
    )


def display_intent(
    intent: str,
) -> str:
    return (
        intent
        .replace("_", " ")
        .title()
    )


def rank_predictions(
    probabilities,
    classes,
    intent_to_category,
    top_k=3,
):
    top_indices = (
        probabilities
        .argsort()[-top_k:][::-1]
    )

    return [
        {
            "intent": classes[index],
            "category": (
                intent_to_category[
                    classes[index]
                ]
            ),
            "confidence": (
                float(
                    probabilities[index]
                )
                * 100
            ),
        }
        for index in top_indices
    ]


def predict_top3(
    text,
    embedding_model,
    classifier,
    intent_to_category,
):
    embedding = (
        embedding_model.encode(
            [text],
            convert_to_numpy=True,
        )
    )

    probabilities = (
        classifier.predict_proba(
            embedding
        )[0]
    )

    return rank_predictions(
        probabilities,
        classifier.classes_,
        intent_to_category,
        top_k=3,
    )
