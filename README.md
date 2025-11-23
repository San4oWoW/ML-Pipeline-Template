# ML Pipeline Template (DVC + MLflow)

Проект демонстрирует минимальный, но полностью воспроизводимый ML‑pipeline с использованием **DVC**, **MLflow**, **Git** и **Docker**.

## Цель проекта

* Воспроизводимый ML‑pipeline
* Отслеживание версий данных через DVC
* Логирование экспериментов в MLflow
* Автоматическая подготовка данных, обучение и оценка модели
* Простая среда запуска через Docker

---

## Структура проекта

```
Ml Pipeline Template/
│
├── data/
│   ├── raw/                # сырой датасет (CSV не хранится в git, только *.dvc)
│   └── processed/          # train/test (генерируются prepare.py)
│
├── src/
│   ├── prepare.py          # сплит данных
│   ├── train.py            # обучение + MLflow
│   └── evaluate.py         # оценка и метрики
│
├── models/                 # модель (не хранится в git)
├── metrics/                # метрики (не хранятся в git)
│
├── dvc.yaml                # описание стадий пайплайна
├── dvc.lock                # зафиксированные версии артефактов
├── params.yaml             # параметры обучения
├── requirements.txt        # зависимости
├── Dockerfile              # контейнеризация
├── mlflow.db               # база MLflow (локальная)
└── README.md
```

---

## Запуск проекта (локально)

Все команды выполнять из корня проекта.

### 1. Установка зависимостей

```
pip install -r requirements.txt
```

### 2. Скачивание данных DVC

(если CSV отсутствует)

```
dvc pull
```

### 3. Запуск пайплайна

```
dvc repro
```

DVC выполнит стадии:

1. **prepare**
2. **train**
3. **evaluate**

---

## Запуск MLflow UI

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Открыть UI:

```
http://127.0.0.1:5000
```

В UI можно смотреть:

* параметры
* метрики
* сохранённую модель

---

## Запуск с Docker

### 1. Собрать контейнер

```
docker build -t ml-pipeline .
```

### 2. Запустить контейнер

```
docker run -p 5000:5000 ml-pipeline
```

MLflow UI будет доступен по адресу:

```
http://localhost:5000
```

---

## Параметры (params.yaml)

```
split:
  test_size: 0.2
  random_state: 42

train:
  model_type: "logreg"
  random_state: 42
  n_estimators: 100
```

---


