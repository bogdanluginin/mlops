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
    """Обчислення метрик якості моделі: RMSE та R²"""
    rmse = root_mean_squared_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, r2


def main():
    # Налаштування аргументів командного рядка
    parser = argparse.ArgumentParser(
        description="Навчання моделі RandomForestRegressor для Bike Sharing Demand"
    )
    parser.add_argument(
        "data_folder",
        type=str,
        help="Шлях до папки з підготовленими даними (data/prepared)"
    )
    parser.add_argument(
        "models_folder",
        type=str,
        help="Шлях до папки для збереження моделей (наприклад, models)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Максимальна глибина дерев для RandomForest"
    )
    args = parser.parse_args()

    data_folder = args.data_folder
    models_folder = args.models_folder

    # Створення папки models, якщо її немає
    os.makedirs(models_folder, exist_ok=True)

    # Завантаження вже підготовлених даних
    print(f"Завантажую дані з {data_folder}...")
    train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))

    # Розділення на ознаки (X) та цільову змінну (y)
    if 'count' not in train_df.columns:
        raise ValueError("Цільова змінна 'count' не знайдена у train.csv")

    X_train = train_df.drop(columns=['count'])
    y_train = train_df['count']
    X_test = test_df.drop(columns=['count'])
    y_test = test_df['count']

    # Ініціалізація MLflow — встановлення назви експерименту
    mlflow.set_experiment("bike-sharing-demand")

    with mlflow.start_run():
        max_depth = args.max_depth

        # Логування гіперпараметрів
        mlflow.log_param("max_depth", max_depth)

        # Встановлення тегів для ідентифікації запуску
        mlflow.set_tag("author", "azzasel")
        mlflow.set_tag("model_type", "RandomForestRegressor")

        # Навчання моделі
        print(f"Навчаю модель з max_depth={max_depth}...")
        model = RandomForestRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Передбачення на тренувальній та тестовій вибірках
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Обчислення метрик якості
        train_rmse, train_r2 = eval_metrics(y_train, train_preds)
        test_rmse, test_r2 = eval_metrics(y_test, test_preds)

        print(f"Метрики на train — RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"Метрики на test  — RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        # Логування метрик у MLflow
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        # Збереження моделі в MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Модель збережено в MLflow.")

        # Збереження моделі локально за допомогою joblib
        model_path = os.path.join(models_folder, "rf_model.pkl")
        joblib.dump(model, model_path)
        print(f"Модель збережено локально: {model_path}")

        # Побудова графіку важливості ознак (Feature Importance)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_train.columns

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=importances[indices],
            y=[features[i] for i in indices],
            palette="viridis"
        )
        plt.title('Feature Importances — RandomForest')
        plt.xlabel('Важливість')
        plt.ylabel('Ознаки')
        plt.tight_layout()

        # Збереження та логування графіку як артефакту MLflow
        fig_path = "feature_importance.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)
        print(f"Графік важливості ознак збережено як артефакт: {fig_path}")


if __name__ == "__main__":
    main()