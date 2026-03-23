# ==========================================
# Stage 1 (Builder): Встановлення залежностей
# ==========================================
FROM python:3.10 AS builder

# Встановлюємо робочу директорію
WORKDIR /app

# Створюємо віртуальне середовище, щоб легко скопіювати всі пакети відразу
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копіюємо лише список залежностей для оптимізації кешування шарів Docker
COPY requirements.txt .

# Встановлюємо залежності без збереження кешу pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2 (Runner): Легковаговий фінальний образ
# ==========================================
FROM python:3.10-slim AS runner

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо ВЖЕ ВСТАНОВЛЕНІ пакети (віртуальне середовище) з builder-стадії
COPY --from=builder /opt/venv /opt/venv

# Переконуємось, що python і pip будуть використовуватись з віртуального середовища
ENV PATH="/opt/venv/bin:$PATH"

# Копіюємо код проєкту та конфігурації
COPY src/ src/
COPY config/ config/
COPY dvc.yaml .

# Вказуємо стандартну команду для запуску (за потребою)
# Наприклад: CMD ["python", "src/optimize.py"]
