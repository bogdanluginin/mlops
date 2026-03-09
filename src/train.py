import pandas as pd
import numpy as np
import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

def eval_metrics(actual, pred):
    rmse = root_mean_squared_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, r2

def main():
    parser = argparse.ArgumentParser(description="Navchannja modeli RandomForestRegressor dlya Bike Sharing Demand")
    parser.add_argument("data_folder", type=str, help="Shlyah do papky z pidgotovlenymy danymy (data/prepared)")
    parser.add_argument("models_folder", type=str, help="Shlyah do papky dlya zberezhennya modeley (np. models)")
    parser.add_argument("--max_depth", type=int, default=10, help="Maksymalna glybyna derev dlya RandomForest")
    args = parser.parse_args()

    data_folder = args.data_folder
    models_folder = args.models_folder
    
    # Stvorennya papky models, yaksho jiyi nemaye
    os.makedirs(models_folder, exist_ok=True)

    # Zavantazhennya vze pidgotovlenyh danyh
    print(f"Zavantazhennya danyh z {data_folder}...")
    train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))

    # Rozdilennya na X ta y
    if 'count' not in train_df.columns:
        raise ValueError("Tsilyova zminna 'count' ne znaydena u train.csv")
    
    X_train = train_df.drop(columns=['count'])
    y_train = train_df['count']
    
    X_test = test_df.drop(columns=['count'])
    y_test = test_df['count']

    # Inicializatsia MLflow
    mlflow.set_experiment("bike-sharing-demand")

    with mlflow.start_run():
        max_depth = args.max_depth
        
        # Loguvannya parametriv
        mlflow.log_param("max_depth", max_depth)
        
        # Vstanovlennya tegiv
        mlflow.set_tag("author", "azzasel")
        mlflow.set_tag("model_type", "RandomForestRegressor")

        print(f"Navchannya modeli z max_depth={max_depth}...")
        model = RandomForestRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Peredbachennya na train ta test
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Obchyslennya metryk
        train_rmse, train_r2 = eval_metrics(y_train, train_preds)
        test_rmse, test_r2 = eval_metrics(y_test, test_preds)

        print(f"Metryky na train - RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"Metryky na test - RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        # Loguvannya metryk v MLflow
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        # Zberigannya modeli v MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Model zberezheno v MLflow.")
        
        # Zberigannya modeli lokaly'no za dopomogoyu joblib
        model_path = os.path.join(models_folder, "rf_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model zberezheno lokal'no: {model_path}")

        # Pobudova grafiku Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_train.columns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette="viridis")
        plt.title('Feature Importances - RandomForest')
        plt.xlabel('Avaga')
        plt.ylabel('Oznaky')
        plt.tight_layout()
        
        # Zberezhennya ta loguvannya grafiku yak artefaktu
        fig_path = "feature_importance.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)
        
        print(f"Grafik oznak zberezheno jak artefakt: {fig_path}")

if __name__ == "__main__":
    main()
