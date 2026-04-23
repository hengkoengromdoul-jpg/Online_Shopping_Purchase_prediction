import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    df = pd.read_excel("online_shopping_dataset.csv.xlsx")

    X = df[["price", "discount", "brand", "rating"]]
    y = df["buy"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    train_model()