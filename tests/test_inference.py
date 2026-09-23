import numpy as np

from src.inference import (
    display_intent,
    load_classifier_artifacts,
    rank_predictions,
)


def test_classifier_artifacts_load():
    classifier, mapping = (
        load_classifier_artifacts()
    )

    assert hasattr(
        classifier,
        "predict_proba",
    )

    assert hasattr(
        classifier,
        "classes_",
    )

    assert len(
        classifier.classes_
    ) == 27

    assert set(
        classifier.classes_
    ).issubset(
        mapping.keys()
    )


def test_embedding_dimension():
    classifier, _ = (
        load_classifier_artifacts()
    )

    assert (
        classifier.n_features_in_
        == 384
    )


def test_display_intent():
    assert (
        display_intent(
            "forgot_password"
        )
        == "Forgot Password"
    )


def test_rank_predictions():
    classes = np.array(
        [
            "intent_a",
            "intent_b",
            "intent_c",
            "intent_d",
        ]
    )

    probabilities = np.array(
        [
            0.10,
            0.50,
            0.30,
            0.10,
        ]
    )

    mapping = {
        "intent_a": "A",
        "intent_b": "B",
        "intent_c": "C",
        "intent_d": "D",
    }

    results = rank_predictions(
        probabilities,
        classes,
        mapping,
        top_k=3,
    )

    assert len(results) == 3

    assert (
        results[0]["intent"]
        == "intent_b"
    )

    assert (
        results[1]["intent"]
        == "intent_c"
    )

    assert (
        results[0]["confidence"]
        == 50.0
    )


def test_classifier_probability_shape():
    classifier, _ = (
        load_classifier_artifacts()
    )

    sample = np.zeros(
        (
            1,
            classifier.n_features_in_,
        )
    )

    probabilities = (
        classifier.predict_proba(
            sample
        )
    )

    assert probabilities.shape == (
        1,
        27,
    )

    assert abs(
        probabilities.sum()
        - 1.0
    ) < 1e-6
