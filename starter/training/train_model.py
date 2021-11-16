# Script to train machine learning model.
import joblib
from sklearn.model_selection import train_test_split

# Add the necessary imports for the training code.
from ml.data import process_data, load_data
from ml.model import train_model, compute_model_metrics, inference, save_model_artifact
# Add code to load in the data.
data = load_data()
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.

model = train_model(X_train, y_train)
save_model_artifact(model, "census_clf.joblib", path="../model")
save_model_artifact(encoder, "census_encoder.joblib", path="../model")

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(precision, recall, fbeta)
