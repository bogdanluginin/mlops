import os
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator

# Airflow 2.3+ deprecated DummyOperator in favor of EmptyOperator. 
# Ми зробимо гнучкий імпорт для зворотної сумісності.
try:
    from airflow.operators.empty import EmptyOperator as DummyOperator
except ImportError:
    from airflow.operators.dummy import DummyOperator

def evaluate_metrics(**kwargs):
    """
    Зчитує metrics.json та маршрутизує DAG залежно від значення RMSE.
    """
    metrics_path = "metrics.json" # Шлях залежить від того, де Airflow worker запускає код!
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}. Stopping pipeline.")
        return 'stop_pipeline'
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    rmse = metrics.get("rmse", float('inf'))
    print(f"Current RMSE: {rmse}")
    
    # Логіка Quality Gate
    if rmse < 50.0:
        return 'register_model'
    else:
        return 'stop_pipeline'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='bike_sharing_pipeline',
    default_args=default_args,
    description='MLOps Pipeline with DVC and Quality Gates',
    schedule_interval=None, # Ручний запуск
    catchup=False,
    tags=['mlops', 'bike_sharing']
) as dag:

    # 1. Підготовка даних
    prepare_data = BashOperator(
        task_id='prepare_data',
        # Команда запускається всередині робочої папки (переконайтесь, що Airflow налаштований)
        bash_command='dvc repro stage:prepare ' 
    )

    # 2. Тренування моделі
    train_model = BashOperator(
        task_id='train_model',
        bash_command='dvc repro stage:train '
    )

    # 3. Branching Operator для перевірки Quality Gate
    check_metrics = BranchPythonOperator(
        task_id='check_metrics',
        python_callable=evaluate_metrics
    )

    # 4. Dummy оператори для гілок
    register_model = DummyOperator(
        task_id='register_model'
    )

    stop_pipeline = DummyOperator(
        task_id='stop_pipeline'
    )

    # 5. Прописуємо граф залежностей (DAG dependencies)
    prepare_data >> train_model >> check_metrics >> [register_model, stop_pipeline]
