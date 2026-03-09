import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

def eval_metrics(actual, pred):
    rmse = root_mean_squared_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, r2

def main():
    parser = argparse.ArgumentParser(description="Navchannja modeli RandomForestRegressor dlya Bike Sharing Demand")
    parser.add_argument("--max_depth", type=int, default=10, help="Maksymalna glybyna derev dlya RandomForest")
    args = parser.parse_args()

    # Zavantazhennya danyh
    print("Zavantazhennya danyh...")
    df = pd.read_csv("data/raw/train.csv")

    # Bazaova peredobrobka (feature engineering)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month

    # Vydalennya zai'vyh kolonok dlya unyknennya vytoku danyh
    df = df.drop(columns=['datetime', 'casual', 'registered'])

    # Rozdilennya na X ta y
    X = df.drop(columns=['count'])
    y = df['count']

    # Rozbyttja na train ta test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

        # Zberigannya modeli
        mlflow.sklearn.log_model(model, "model")
        print("Model zberezheno v MLflow.")

        # Pobudova grafiku Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns
        
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
