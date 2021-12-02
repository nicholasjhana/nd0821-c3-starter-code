# Script to train machine learning model.
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the training code.
from ml.data import process_data, load_data
from ml.model import train_model, compute_model_metrics, inference, save_model_artifact
from ml.slices import slice_performance
# Add code to load in the data.
data = load_data()
print("data shape")
print(data.shape)
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

print(lb.get_params())
print(lb.transform([">50K"]))
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
print("X_train shape")
print(X_train.shape)

print("X_test shape")
print(X_test.shape)

model = train_model(X_train, y_train)
save_model_artifact(model, "census_clf.joblib", path="./model")
save_model_artifact(encoder, "census_encoder.joblib", path="./model")

preds = inference(model, X_test)
print(preds)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Model results. Precision: {precision} Recall: {recall} FBeta: {fbeta}")

print("-------------------")
print("Computing slice performance")

#Construct bias from test dataset X_test, preds, y


file_name = "slice_output.txt"

with open(file_name, "w+") as file:
    # file.write("Feature Name, Categorical Value, Precision, Recall, FBeta")
    for feat in cat_features:
        slice_values = slice_performance(test, preds, feat)

        for key, val in slice_values.items():
            line = f"\n{feat}, {key}, {val[0]}, {val[1]}, {val[2]}"
            file.write(line)
    file.close()


print(f"Slice performance output to {file_name}")