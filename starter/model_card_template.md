# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
N.Shaw created the model. It is a random forest classifier from scikit-learn 1.0.1. Unless otherwise listed below hyperparameters are default.
Modified hyperparameters are:
- n_estimators=18 
- max_depth=7
- random_state=99

## Intended Use
Model is for predicting if a citizen's income is >$50K/yr. Prospective uses are for example loan application evaluations, or demographic studies.

## Training Data
The data was obtained from the UCI machine learning repository under the name [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income)
The target class is provided within the dataset under the "salary" column header. It contains a binary class denoted as "<=50K" or ">50K".

The data set contains 32562 entries that were split 80-20 into a training and testing sets. 

The training set contained 26050 samples. In preparation for training the data set was lightly cleaned by removing spaces from the column headers, and within any categorical columns.
As part of preprocessing, categorical features were one hot encoded and the target feature was binarized.

## Evaluation Data
Evaluation data comprised of the 20% of data not used in training, a total of 6512 samples. It was preprocessed in the same method as training data using the encoder and label binarizer artifacts produced.

## Metrics
Three metrics were used to evaluate the model's performance. Each metric is listed below with its performance. 

| Metric| Training | Testing |
|-------|-----|---------|
| precision | 0.97 | 0.77    | 
| recall | 0.89 | 0.62    |
| fbeta | 0.93 | 0.69    |

## Ethical Considerations

## Caveats and Recommendations
