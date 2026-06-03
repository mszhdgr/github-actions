from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature


df = pd.read_csv("data/calif_housing.csv")

x = df.drop("MedHouseVal", axis=1)
labels = df["MedHouseVal"].apply(lambda x: x * 100000)
y = labels.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    R2Score = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    signature = infer_signature(X_train, model_output=model.predict(X_test))

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("r2_score", R2Score)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model, name="Rfr_model")

    joblib.dump(model, "models/Rfr_model.pkl")

    print("Model Training Complete...\n")
    print(f"R2_score: {R2Score:.4f} || mae: {mae:.4f} || mse: {mse:.4f} ")