# AI Support Ticket Classifier

An NLP machine learning application that automatically classifies customer support messages by **intent** and **category**.

The project compares traditional TF-IDF classification with semantic sentence embeddings and includes a Streamlit web application for real-time predictions.

## Project Overview

Customer support teams receive large volumes of messages that need to be routed to the correct department.

This project builds a multiclass NLP classifier capable of identifying the customer's intent from a support message.

Example:

```text
"I forgot my password"
```

Prediction:

```text
Category: ACCOUNT
Intent: recover_password
Confidence: 98%+
```

The final system also detects uncertain predictions and recommends human review when confidence is low.

## Dataset

The project uses the Bitext Customer Support dataset.

Dataset size:

- 26,872 customer messages
- 27 intents
- 11 broader categories
- No missing values

Main columns:

- `instruction` — customer message
- `intent` — specific support intent
- `category` — broader support category
- `response` — example support response
- `flags` — metadata

The target variable used in this project is `intent`.

## Exploratory Analysis

The dataset contains 27 relatively balanced intent classes.

Examples include:

- `recover_password`
- `track_order`
- `cancel_order`
- `get_refund`
- `payment_issue`
- `change_shipping_address`
- `contact_human_agent`
- `check_invoice`
- `create_account`

Because the dataset is well balanced, Macro F1 is an appropriate metric for comparing models.

## Model 1 — TF-IDF + Logistic Regression

The first baseline model used:

- TF-IDF vectorization
- Unigrams and bigrams
- Logistic Regression

Test performance:

```text
Accuracy: 99.40%
Macro F1: 99.41%
```

5-fold cross-validation:

```text
Macro F1 mean: 99.25%
Standard deviation: 0.047%
```

## Model 2 — TF-IDF + Linear SVM

A Linear Support Vector Machine was then tested using the same TF-IDF representation.

Test performance:

```text
Accuracy: 99.52%
Macro F1: 99.52%
```

5-fold cross-validation:

```text
Macro F1 mean: 99.43%
Standard deviation: 0.104%
```

The Linear SVM achieved the strongest traditional test metrics.

## Probability Calibration

Because `LinearSVC` does not natively output class probabilities, probability calibration was tested using `CalibratedClassifierCV`.

Calibrated SVM performance:

```text
Accuracy: 99.48%
Macro F1: 99.48%
```

This allowed meaningful probability estimates while maintaining almost identical classification performance.

## Semantic NLP Model

Traditional TF-IDF models rely heavily on the presence of specific words and phrases.

To improve generalization to unseen wording, a semantic model was tested using:

- `all-MiniLM-L6-v2`
- Sentence Transformers
- 384-dimensional sentence embeddings
- Logistic Regression classifier

Performance:

```text
Accuracy: 99.40%
Macro F1: 99.40%
```

Although the Linear SVM produced slightly higher test metrics, the semantic model performed better on several manually written messages outside the original dataset.

For example:

```text
"My order has not arrived yet"
```

TF-IDF SVM prediction:

```text
create_account
```

Semantic model prediction:

```text
delivery_period
```

The semantic model was therefore selected as the final production model because of its ability to generalize based on sentence meaning rather than only lexical similarity.

## Confidence Handling

The final application returns:

- predicted intent
- predicted category
- model probability
- top 3 predictions

Predictions are divided into three confidence levels:

```text
High confidence:   >= 70%
Medium confidence: 40%–70%
Low confidence:    < 40%
```

Low-confidence predictions are flagged for human review.

This is particularly useful for ambiguous messages such as:

```text
"I need help"
```

where the customer has not provided enough information for a reliable automatic decision.

The confidence thresholds are product rules rather than guarantees of correctness.

## Streamlit Application

The project includes an interactive Streamlit interface.

Users can:

- enter a customer support message
- receive the predicted intent
- see the broader support category
- inspect model confidence
- view the top 3 predictions
- identify messages that may require human review

### High-confidence prediction

![High confidence prediction](images/App%20High%20Confidence.png)

### Low-confidence prediction

![Low confidence prediction](images/App%20Low%20Confidence.png)

## Project Structure

```text
nlp-support-classifier/
├── app/
│   └── app.py
├── data/
│   └── raw/
│       └── customer_support.csv
├── images/
│   ├── App High Confidence.png
│   └── App Low Confidence.png
├── models/
│   ├── semantic_intent_classifier.joblib
│   └── intent_to_category.joblib
├── notebooks/
│   └── exploration.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Run Locally

Clone the repository.

Create and activate a Python virtual environment.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

## Technologies

- Python
- pandas
- scikit-learn
- Sentence Transformers
- Hugging Face
- TF-IDF
- Logistic Regression
- Linear SVM
- Probability Calibration
- Streamlit
- joblib
- Jupyter Notebook

## Key Lessons

This project demonstrates several important NLP concepts:

- multiclass text classification
- TF-IDF vectorization
- sentence embeddings
- semantic similarity
- model comparison
- cross-validation
- probability calibration
- confidence-based human review
- model persistence
- deployment of an NLP model through Streamlit

A key finding was that the model with the highest standard test score was not necessarily the best model for unseen real-world phrasing.

Testing manually written examples revealed limitations that were not obvious from the test-set metrics alone.

## Future Improvements

Possible future improvements include:

- fine-tuning a transformer directly for intent classification
- expanding the training data with more varied real-world messages
- multilingual support
- out-of-distribution detection
- automatic ticket routing
- API deployment
- model monitoring
- feedback-based retraining
- automated testing and CI/CD
