import pandas as pd

def rescale (x):
    a = 0.165 / 0.37
    b = (1 - 0.165) / (1 - 0.37)
    return a * x / (a * x + b * (1 - x))


model1_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/rescale_0.35833_preds_2018-02-20-23-28-20_0.34951.csv"
model2_path = "/Users/skelter/Desktop/QuoraDupDetection/Keras/rescale_submission_200218_0.35347_0.31943.csv"

model1_pred = pd.read_csv(model1_path)
model2_pred = pd.read_csv(model1_path)

submission = pd.DataFrame(
    {
        "is_duplicate": "%.5f" % rescale((model1_pred["is_duplicate"] + model2_pred["is_duplicate"])/2),
        "test_id": model1_pred["test_id"]
    })

submission.to_csv("ensemble.csv", index=False)