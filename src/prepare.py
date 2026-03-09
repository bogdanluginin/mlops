import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split


def process_data(input_file, output_folder):
    """Завантаження та попередня обробка даних"""

    print(f"Завантажую дані з {input_file}...")

    # Завантаження даних за допомогою pandas
    df = pd.read_csv(input_file)

    # Feature engineering
    print("Виконую feature engineering...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month

    # Видалення зайвих колонок для уникнення витоку даних
    cols_to_drop = ['datetime', 'casual', 'registered']

    # Перевіряємо чи є ці колонки перед видаленням
    # (щоб не було помилки при запуску на test.csv)
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Перевірка чи маємо цільову змінну 'count' (для raw/train.csv),
    # або це raw/test.csv де її немає
    if 'count' in df.columns:
        print("Розбиваю дані на train та test...")

        # Розбиваємо датафрейм на тренувальну та тестову вибірки
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Збереження оброблених даних у папку виводу
        os.makedirs(output_folder, exist_ok=True)
        train_path = os.path.join(output_folder, "train.csv")
        test_path = os.path.join(output_folder, "test.csv")

        print(f"Зберігаю оброблені дані у {train_path} та {test_path}...")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

    else:
        # Якщо це test.csv (з data/raw) — просто обробляємо та зберігаємо
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "test_features.csv")

        print(f"Зберігаю оброблену тестову вибірку у {out_path}...")
        df.to_csv(out_path, index=False)

    print("Готово!")


if __name__ == "__main__":
    # Скрипт приймає два аргументи через sys.argv:
    # 1 — шлях до вхідного файлу
    # 2 — шлях до папки виводу
    if len(sys.argv) != 3:
        print("Використання: python src/prepare.py <шлях_до_вхідного_файлу> <шлях_до_папки_виводу>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_folder_path = sys.argv[2]

    process_data(input_file_path, output_folder_path)