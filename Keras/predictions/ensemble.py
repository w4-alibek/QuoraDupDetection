import pandas as pd

def rescale (x):
    a = 0.165 / 0.37
    b = (1 - 0.165) / (1 - 0.37)
    return a * x / (a * x + b * (1 - x))


model1_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/predictions/preds_0.31921.csv"
model2_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/predictions/preds_0.32347.csv"
model3_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/predictions/preds_0.32752.csv"
model4_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/predictions/preds_0.32783.csv"
model5_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/predictions/preds_0.35347.csv"
model6_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/predictions/preds_0.36183.csv"
#model7_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/predictions/rescale_0.36183_r0.32267.csv"

model1_pred = pd.read_csv(model1_path)
model2_pred = pd.read_csv(model2_path)
model3_pred = pd.read_csv(model3_path)
model4_pred = pd.read_csv(model4_path)
model5_pred = pd.read_csv(model5_path)
model6_pred = pd.read_csv(model6_path)
#model7_pred = pd.read_csv(model7_path)

submission = pd.DataFrame(
    {
        "is_duplicate": (
                                model1_pred["is_duplicate"]
                                + model2_pred["is_duplicate"]
                                + model3_pred["is_duplicate"]
                                + model4_pred["is_duplicate"]
                                + model5_pred["is_duplicate"]
                                + model6_pred["is_duplicate"]
#                                + model7_pred["is_duplicate"]
                        )/6.000,
        "test_id": model1_pred["test_id"]
    })

submission.to_csv("ensemble_before_rescale.csv", index=False)