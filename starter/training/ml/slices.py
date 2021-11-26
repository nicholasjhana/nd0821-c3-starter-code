import pandas as pd
from ml.model import compute_model_metrics




def slice_performance(X, preds, category):
    """ Function for calculating model performance on on fixed values of the feautre.
     Will calculate model metrics for each value in the categroiacal feature.

     Inputs
    ------
    df : pd.DataFrame
        Preprocessed dataframe containing testing or training data.
    model : sklearn Model.
        A trained model ready for inference.
    feature : string
        Categorical feature to fix value.
    label: string
        Target label. default is "salary"

    Outputs
    ------

     """

    X["preds"] = preds

    X["actual"] = X["salary"].replace({
        "<=50K": 0,
        ">50K": 1
    })

    slice_dict = dict()
    for val in X[category].unique():
        df_slice = X[X[category] == val]
        precision, recall, fbeta = compute_model_metrics(df_slice["actual"].values,
                                                         df_slice["preds"].values)
        slice_dict[val] = [precision, recall, fbeta]
        # print(f"{category} - {val} - Precision: {precision} Recall: {recall} FBeta: {fbeta}")

    return slice_dict