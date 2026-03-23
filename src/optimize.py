import hydra
from omegaconf import DictConfig
import optuna
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def optimize(cfg: DictConfig):
    # Set MLflow experiment
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # Load data
    data_path = cfg.data.processed_path
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Please make sure the data is prepared.")
        return

    # Assume the target column is the last one if not explicitly specified
    # Typical target columns for Bike Sharing Demand are 'count' or 'cnt'
    target_col = 'count' if 'count' in df.columns else ('cnt' if 'cnt' in df.columns else df.columns[-1])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    def objective(trial):
        # Open a child run for each trial
        with mlflow.start_run(nested=True):
            # Suggest hyperparameters
            max_depth = trial.suggest_int(
                "max_depth", 
                cfg.hpo.search_space.max_depth.low, 
                cfg.hpo.search_space.max_depth.high
            )
            n_estimators = trial.suggest_int(
                "n_estimators", 
                cfg.hpo.search_space.n_estimators.low, 
                cfg.hpo.search_space.n_estimators.high
            )
            
            # Log params for the child run
            mlflow.log_params({
                "max_depth": max_depth,
                "n_estimators": n_estimators
            })
            
            model = RandomForestRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            
            # Evaluate using cross-validation
            neg_rmse_scores = cross_val_score(
                model, X, y, cv=3, scoring="neg_root_mean_squared_error"
            )
            rmse = -neg_rmse_scores.mean()
            
            # Log metric for the child run
            mlflow.log_metric("rmse", rmse)
            
            return rmse

    # Start parent run
    with mlflow.start_run(run_name="Optuna_RandomForest_Optimization"):
        # Initialize sampler
        sampler_name = cfg.hpo.sampler
        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)
            
        # Create Optuna study
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        
        # Run optimization
        study.optimize(objective, n_trials=cfg.hpo.n_trials)
        
        # Логуємо найкращі параметри та метрику у батьківський run
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse", study.best_value)
        
        # Зберігаємо метрики у файл для тестування (Quality Gate)
        import json
        with open("metrics.json", "w") as f:
            json.dump({"rmse": study.best_value}, f)
        
        print("\nОптимізація завершена.")
        print("Best trial:")
        print(f"  RMSE: {study.best_value:.4f}")
        print("  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    optimize()
