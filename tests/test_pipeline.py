import pandas as pd
import json
import os

def test_data_schema():
    """
    Pre-train тест: Перевірка схеми та якості сирих даних.
    """
    data_path = "data/raw/train.csv"
    
    # Перевіряємо, чи існує файл
    assert os.path.exists(data_path), f"Raw data file not found at {data_path}"
    
    df = pd.read_csv(data_path)
    
    # 1. Перевірка наявності ключових колонок
    required_columns = ['datetime', 'temp', 'count']
    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' is missing in the dataset"
        
    # 2. Перевірка на відсутність пропущених значень (відомо, що Bike Sharing не має пропусків)
    missing_values = df.isnull().sum().sum()
    assert missing_values == 0, f"Dataset contains {missing_values} missing values (NaNs)"

def test_quality_gate():
    """
    Post-train тест (Quality Gate): Перевірка якості моделі за метрикою RMSE.
    """
    metrics_path = "metrics.json"
    
    # Перевіряємо наявність файлу з метриками
    assert os.path.exists(metrics_path), f"Metrics file {metrics_path} not found"
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    # Перевіряємо, чи збережена метрика rmse
    assert "rmse" in metrics, "Key 'rmse' is missing in metrics.json"
    
    rmse = metrics["rmse"]
    
    # Quality Gate (поріг 50.0)
    threshold = 50.0
    assert rmse < threshold, f"Model failed Quality Gate! RMSE {rmse:.4f} is >= {threshold}"
