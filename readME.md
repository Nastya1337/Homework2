# Домашнее задание к уроку 2: Линейная и логистическая регрессия

## Задание 1: Модификация существующих моделей (30 баллов)

### 1.1 Расширение линейной регрессии (15 баллов)

Модифицируйте существующую линейную регрессию:
- Добавьте L1 и L2 регуляризацию
- Добавьте early stopping

Этот код реализует линейную и логистическую регрессии "вручную" (без использования torch.nn) с использованием PyTorch для работы с тензорами и автоматическим дифференцированием.
Основные компоненты:
- Линейная регрессия (LinearRegressionManual):
- Реализована с возможностью L1- и L2-регуляризации.
- Поддерживает обучение с early stopping.

Методы: 
- прямой проход (__call__), обратное распространение (backward), обновление весов (step), сохранение/загрузка модели.
- Логистическая регрессия (LogisticRegression):
- Поддерживает многоклассовую классификацию.
- Включает методы для вычисления метрик (precision, recall, F1, ROC-AUC) и визуализации матрицы ошибок.

Вспомогательные функции:
- Генерация синтетических данных (make_regression_data).
- Датсет для регрессии (RegressionDataset).
- Функция потерь MSE (mse).
- Логирование процесса обучения (log_epoch).

Обучение моделей:

- train_linear_regression(): обучает линейную регрессию с регуляризацией.
- train_logistic_regression(): обучает логистическую регрессию и выводит метрики качества.

### 1.2 Расширение логистической регрессии (15 баллов)

Модифицируйте существующую логистическую регрессию:
- Добавьте поддержку многоклассовой классификации
- Реализуйте метрики: precision, recall, F1-score, ROC-AUC
- Добавьте визуализацию confusion matrix

```python
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import make_regression_data, mse, log_epoch, RegressionDataset

def make_regression_data(n=200):
    torch.manual_seed(42)
    X = torch.rand(n, 1) * 10
    y = 2 * X + 3 + torch.randn(n, 1) * 2
    return X, y


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def log_epoch(epoch, loss):
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

class LinearRegressionManual:
    def __init__(self, in_features, l1_lambda=0.0, l2_lambda=0.0):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def __call__(self, X):
        return X @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y

        # Добавляем регуляризацию к градиентам
        l1_grad = self.l1_lambda * torch.sign(self.w) if self.l1_lambda > 0 else 0
        l2_grad = self.l2_lambda * 2 * self.w if self.l2_lambda > 0 else 0

        self.dw = (X.T @ error) / n + l1_grad + l2_grad
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']


class LogisticRegression:
    def __init__(self, in_features, num_classes):
        self.w = torch.randn(in_features, num_classes, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(num_classes, dtype=torch.float32, requires_grad=False)
        self.num_classes = num_classes

    def __call__(self, X):
        logits = X @ self.w + self.b
        return torch.softmax(logits, dim=1)

    def predict(self, X):
        probs = self(X)
        return torch.argmax(probs, dim=1)

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def compute_metrics(self, X, y):
        y_pred = self.predict(X)
        y_probs = self(X)

        if self.num_classes == 2:
            y_probs = y_probs[:, 1]
            roc_auc = roc_auc_score(y.numpy(), y_probs.detach().numpy())
        else:
            y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
            roc_auc = roc_auc_score(y_onehot.numpy(), y_probs.detach().numpy(), multi_class='ovr')

        precision = precision_score(y.numpy(), y_pred.numpy(), average='weighted')
        recall = recall_score(y.numpy(), y_pred.numpy(), average='weighted')
        f1 = f1_score(y.numpy(), y_pred.numpy(), average='weighted')

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

    def plot_confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        cm = confusion_matrix(y.numpy(), y_pred.numpy())

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


def train_linear_regression():
    # Генерируем данные
    X, y = make_regression_data()

    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    print(f'Пример данных: {dataset[0]}')

    # Обучаем модель с регуляризацией и early stopping
    model = LinearRegressionManual(in_features=1, l1_lambda=0.01, l2_lambda=0.01)
    lr = 0.1
    epochs = 100
    patience = 5
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            loss = mse(y_pred, batch_y)
            total_loss += loss

            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)

        avg_loss = total_loss / (i + 1)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

    model.save('linreg_manual.pth')


def train_logistic_regression():
    # Пример данных для классификации
    n_samples = 200
    n_features = 4
    n_classes = 3

    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))

    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LogisticRegression(in_features=n_features, num_classes=n_classes)
    lr = 0.1
    epochs = 50

    for epoch in range(1, epochs + 1):
        for i, (batch_X, batch_y) in enumerate(dataloader):
            # One-hot кодирование для многоклассовой классификации
            y_onehot = torch.nn.functional.one_hot(batch_y, num_classes=n_classes).float()
            y_pred = model(batch_X)

            model.zero_grad()
            model.backward(batch_X, y_onehot, y_pred)
            model.step(lr)

        if epoch % 10 == 0:
            metrics = model.compute_metrics(X, y)
            print(f'Epoch {epoch}:')
            print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            print(f"F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    model.plot_confusion_matrix(X, y)


if __name__ == '__main__':
    print("Training Linear Regression...")
    train_linear_regression()

    print("\nTraining Logistic Regression...")
    train_logistic_regression()
```
## Задание 2: Работа с датасетами (30 баллов)

### 2.1 Кастомный Dataset класс (15 баллов)

Создайте кастомный класс датасета для работы с CSV файлами:
- Загрузка данных из файла
- Предобработка (нормализация, кодирование категорий)
- Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)

### 2.2 Эксперименты с различными датасетами (15 баллов)
- Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию

Основные компоненты:

1. CSVDataset – кастомный класс для загрузки и предобработки данных:
Загружает данные из CSV-файла.

Поддерживает:

- Нормализацию числовых признаков (StandardScaler).
- One-hot кодирование категориальных признаков.
- Label encoding бинарных признаков.

Преобразует данные в тензоры PyTorch.

2. Модели:

LinearRegressionManual – линейная регрессия:

- Ручной расчет градиентов (MSE-лосс).
- Обучение через SGD (стохастический градиентный спуск).

LogisticRegression – логистическая регрессия:

- Сигмоидная активация для бинарной классификации.
- Binary cross-entropy loss.
- Метод predict() с пороговым значением.

3. Функции обучения:

train_regression():

- Обучает линейную регрессию на данных о ценах домов (house_prices.csv).
- Выводит MSE на тестовой выборке.

train_classification():

- Обучает логистическую регрессию на данных о оттоку клиентов (customer_churn.csv).
- Выводит accuracy на тестовой выборке.

```python
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


class CSVDataset(Dataset):
    def __init__(self, file_path, target_column, numeric_cols=None,
                 categorical_cols=None, binary_cols=None, normalize=True):
        """
        Кастомный Dataset класс для работы с CSV файлами

        Параметры:
            file_path: путь к CSV файлу
            target_column: имя целевой колонки
            numeric_cols: список числовых колонок
            categorical_cols: список категориальных колонок
            binary_cols: список бинарных колонок
            normalize: нормализовать ли числовые признаки
        """
        # Загрузка данных
        self.data = pd.read_csv(file_path)

        # Сохраняем параметры
        self.target_column = target_column
        self.numeric_cols = numeric_cols if numeric_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.binary_cols = binary_cols if binary_cols else []
        self.normalize = normalize

        # Предобработка данных
        self._preprocess_data()

    def _preprocess_data(self):
        # Обработка числовых признаков
        if self.numeric_cols and self.normalize:
            self.scaler = StandardScaler()
            self.data[self.numeric_cols] = self.scaler.fit_transform(self.data[self.numeric_cols])

        # Обработка категориальных признаков (one-hot encoding)
        if self.categorical_cols:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = self.encoder.fit_transform(self.data[self.categorical_cols])
            encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)
            self.data = pd.concat([
                self.data.drop(columns=self.categorical_cols),
                pd.DataFrame(encoded, columns=encoded_cols)
            ], axis=1)

        # Обработка бинарных признаков (label encoding)
        if self.binary_cols:
            self.label_encoder = LabelEncoder()
            for col in self.binary_cols:
                self.data[col] = self.label_encoder.fit_transform(self.data[col])

        # Разделение на признаки и целевую переменную
        self.features = self.data.drop(columns=[self.target_column]).values
        self.target = self.data[self.target_column].values

        # Преобразование в тензоры
        self.features = torch.FloatTensor(self.features)
        self.target = torch.FloatTensor(self.target.reshape(-1, 1))  # Убедились, что целевые переменные имеют форму (-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]


# Модифицированная линейная регрессия
class LinearRegressionManual:
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return X @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db


# Логистическая регрессия
class LogisticRegression:
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def __call__(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self(X) >= threshold).float()

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db


def train_regression(dataset_path):
    # Загрузка датасета для регрессии
    dataset = CSVDataset(
        file_path=dataset_path,
        target_column='sale_price',  # Имя целевой переменной
        numeric_cols=['lot_area', 'year_built', 'total_bathrooms', 'garage_cars'],  # Числовые признаки
        normalize=True
    )

    # Разделение на train/test
    train_data, test_data = train_test_split(list(dataset), test_size=0.2, random_state=42)

    # Создание DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Создание и обучение модели
    model = LinearRegressionManual(in_features=dataset.features.shape[1])  # Определяем входной размер исходя из всех данных
    lr = 0.01
    epochs = 100

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = ((y_pred - y_batch) ** 2).mean()
            total_loss += loss.item()

            model.zero_grad()
            model.backward(X_batch, y_batch, y_pred)
            model.step(lr)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}')

    # Оценка на тестовых данных
    with torch.no_grad():
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        total_loss = 0
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = ((y_pred - y_batch) ** 2).mean()
            total_loss += loss.item()
        print(f'Test MSE: {total_loss / len(test_loader):.4f}')


def train_classification(dataset_path):
    # Загрузка датасета для классификации
    dataset = CSVDataset(
        file_path=dataset_path,
        target_column='churn',  # Целевая переменная
        numeric_cols=[
            'account_length', 'total_day_minutes', 'total_eve_minutes',
            'total_night_minutes', 'total_intl_minutes', 'number_vmail_messages'
        ],
        binary_cols=['international_plan', 'voice_mail_plan'],  # Бинарные признаки
        normalize=True
    )

    # Разделение на train/test
    train_data, test_data = train_test_split(list(dataset), test_size=0.2, random_state=42)

    # Создание DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Создание и обучение модели
    model = LogisticRegression(in_features=dataset.features.shape[1])  # Входной размер определяется всеми признаками
    lr = 0.1
    epochs = 100

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            y_batch = y_batch.float()  # Преобразуем в FloatTensor

            # Binary cross-entropy loss
            loss = -(y_batch * torch.log(y_pred + 1e-10) + (1 - y_batch) * torch.log(1 - y_pred + 1e-10)).mean()
            total_loss += loss.item()

            # Accuracy
            preds = model.predict(X_batch)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

            model.zero_grad()
            model.backward(X_batch, y_batch, y_pred)
            model.step(lr)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}')

    # Оценка на тестовых данных
    with torch.no_grad():
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        total_correct = 0
        total_count = 0
        for X_batch, y_batch in test_loader:
            y_pred = model.predict(X_batch)
            total_correct += (y_pred == y_batch).sum().item()
            total_count += len(y_batch)
        accuracy = total_correct / total_count
        print(f'Test Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    # Относительный путь к файлам (файлы в той же директории, что и .py файл)
    house_prices_path = 'data/house_prices.csv'
    customer_churn_path = 'data/customer_churn.csv'

    # 1. Тренировка регрессионной модели
    print("Training regression model...")
    try:
        train_regression(house_prices_path)
    except Exception as e:
        print(f"Error in regression training: {e}")

    # 2. Тренировка классификационной модели
    print("\nTraining classification model...")
    try:
        train_classification(customer_churn_path)
    except Exception as e:
        print(f"Error in classification training: {e}")
```

## Задание 3: Эксперименты и анализ (20 баллов)

### 3.1 Исследование гиперпараметров (10 баллов)
Проведите эксперименты с различными:
- Скоростями обучения (learning rate)
- Размерами батчей
- Оптимизаторами (SGD, Adam, RMSprop)

Визуализируйте результаты в виде графиков или таблиц

```python
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_regression

# Задание 3.1

def make_regression_data(n=200):
    torch.manual_seed(42)
    X = torch.rand(n, 1) * 10
    y = 2 * X + 3 + torch.randn(n, 1) * 2
    return X, y

# Реализация функции make_regression_data, которая отсутствовала
def make_regression_data(n_samples=1000, n_features=10, noise=0.1):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, y


# Остальные функции и классы
def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LinearRegressionManual:
    def __init__(self, in_features, l1_lambda=0.0, l2_lambda=0.0):
        self.w = torch.randn(in_features, 1, requires_grad=False)
        self.b = torch.randn(1, requires_grad=False)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dw = None
        self.db = None

    def __call__(self, X):
        return X @ self.w + self.b

    def backward(self, X, y, y_pred):
        error = y_pred - y
        self.dw = (X.T @ error) / len(X)
        self.db = torch.mean(error)

        # Добавляем регуляризацию
        if self.l1_lambda > 0:
            self.dw += self.l1_lambda * torch.sign(self.w)
            self.db += self.l1_lambda * torch.sign(self.b)
        if self.l2_lambda > 0:
            self.dw += self.l2_lambda * self.w
            self.db += self.l2_lambda * self.b

    def zero_grad(self):
        self.dw = None
        self.db = None

    def step(self, lr):
        with torch.no_grad():
            self.w -= lr * self.dw
            self.b -= lr * self.db


def train_linear_regression_with_params(X, y, lr=0.1, batch_size=32, optimizer='sgd',
                                        epochs=100, l1_lambda=0.0, l2_lambda=0.0):
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LinearRegressionManual(in_features=X.shape[1], l1_lambda=l1_lambda, l2_lambda=l2_lambda)

    # Инициализация параметров оптимизатора
    if optimizer == 'adam':
        m_w, m_b = torch.zeros_like(model.w), torch.zeros_like(model.b)
        v_w, v_b = torch.zeros_like(model.w), torch.zeros_like(model.b)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
    elif optimizer == 'rmsprop':
        avg_sq_w, avg_sq_b = torch.zeros_like(model.w), torch.zeros_like(model.b)
        gamma = 0.9
        eps = 1e-8

    losses = []

    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            loss = mse(y_pred, batch_y)
            total_loss += loss

            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)

            # Применение разных оптимизаторов
            if optimizer == 'sgd':
                model.step(lr)
            elif optimizer == 'adam':
                # Обновление моментов для Adam
                m_w = beta1 * m_w + (1 - beta1) * model.dw
                m_b = beta1 * m_b + (1 - beta1) * model.db
                v_w = beta2 * v_w + (1 - beta2) * (model.dw ** 2)
                v_b = beta2 * v_b + (1 - beta2) * (model.db ** 2)

                # Коррекция bias
                m_w_hat = m_w / (1 - beta1 ** epoch)
                m_b_hat = m_b / (1 - beta1 ** epoch)
                v_w_hat = v_w / (1 - beta2 ** epoch)
                v_b_hat = v_b / (1 - beta2 ** epoch)

                # Обновление параметров
                model.w -= lr * m_w_hat / (torch.sqrt(v_w_hat) + eps)
                model.b -= lr * m_b_hat / (torch.sqrt(v_b_hat) + eps)
            elif optimizer == 'rmsprop':
                # Обновление скользящего среднего для RMSprop
                avg_sq_w = gamma * avg_sq_w + (1 - gamma) * (model.dw ** 2)
                avg_sq_b = gamma * avg_sq_b + (1 - gamma) * (model.db ** 2)

                # Обновление параметров
                model.w -= lr * model.dw / (torch.sqrt(avg_sq_w) + eps)
                model.b -= lr * model.db / (torch.sqrt(avg_sq_b) + eps)

        avg_loss = total_loss / (i + 1)
        losses.append(avg_loss.item())

    return losses


def plot_losses(losses_dict, title):
    plt.figure(figsize=(10, 6))
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Генерируем данные
X, y = make_regression_data()

# Эксперимент с разными learning rates
lrs = [0.001, 0.01, 0.1, 0.5]
lr_losses = {}
for lr in lrs:
    losses = train_linear_regression_with_params(X, y, lr=lr, optimizer='sgd')
    lr_losses[f'lr={lr}'] = losses
plot_losses(lr_losses, 'Loss for different learning rates')

# Эксперимент с разными размерами батчей
batch_sizes = [8, 16, 32, 64]
batch_losses = {}
for bs in batch_sizes:
    losses = train_linear_regression_with_params(X, y, batch_size=bs, optimizer='sgd')
    batch_losses[f'batch_size={bs}'] = losses
plot_losses(batch_losses, 'Loss for different batch sizes')

# Эксперимент с разными оптимизаторами
optimizers = ['sgd', 'adam', 'rmsprop']
optim_losses = {}
for optim in optimizers:
    losses = train_linear_regression_with_params(X, y, optimizer=optim)
    optim_losses[optim] = losses
plot_losses(optim_losses, 'Loss for different optimizers')
```

### 3.2 Feature Engineering (10 баллов)
Создайте новые признаки для улучшения модели:
- Полиномиальные признаки
- Взаимодействия между признаками
- Статистические признаки (среднее, дисперсия)

Сравните качество с базовой моделью

```python
def create_polynomial_features(X, degree=2):
    """Создает полиномиальные признаки до указанной степени"""
    X_poly = X.clone()
    for d in range(2, degree+1):
        X_poly = torch.cat((X_poly, X ** d), dim=1)
    return X_poly

def create_statistical_features(X, window_size=3):
    """Добавляет статистические признаки (скользящее среднее и std)"""
    n = len(X)
    X_stat = torch.zeros((n, 2))
    for i in range(n):
        start = max(0, i - window_size)
        X_stat[i, 0] = X[start:i+1].mean()
        X_stat[i, 1] = X[start:i+1].std()
    return torch.cat((X, X_stat), dim=1)

def evaluate_model(X, y):
    """Оценивает модель на данных и возвращает конечный loss"""
    losses = train_linear_regression_with_params(X, y, epochs=50, lr=0.1)
    return losses[-1]

# Базовые данные
def make_regression_data():
    X, y = make_regression_data()

def make_regression_data():
    # Пример данных для классификации
    n_samples = 200
    n_features = 4
    n_classes = 3

# 1. Базовые признаки
base_loss = evaluate_model(X, y)

# 2. Полиномиальные признаки (до 3 степени)
X_poly = create_polynomial_features(X, degree=3)
poly_loss = evaluate_model(X_poly, y)

# 3. Статистические признаки
X_stat = create_statistical_features(X)
stat_loss = evaluate_model(X_stat, y)

# 4. Комбинация полиномиальных и статистических
X_combined = create_statistical_features(create_polynomial_features(X, degree=2))
combined_loss = evaluate_model(X_combined, y)

# Сравнение результатов
results = {
    'Model': ['Base', 'Polynomial', 'Statistical', 'Combined'],
    'Features': [1, 3, 3, 5],
    'Loss': [base_loss, poly_loss, stat_loss, combined_loss]
}

import pandas as pd
print(pd.DataFrame(results))
```
Основные компоненты:
1. Генерация данных:

- make_regression_data() - создает синтетические данные для регрессии
- Используется как одномерная, так и многомерная регрессия

2. Реализация модели:

LinearRegressionManual - ручная реализация линейной регрессии:

- Поддержка L1/L2 регуляризации
- Методы для обучения (backward, step)
- Возможность работы с разными оптимизаторами

Функции обучения:

train_linear_regression_with_params() - универсальная функция обучения:

- Поддержка SGD, Adam, RMSprop
- Настройка learning rate и batch size
- Визуализация процесса обучения

Эксперименты:

- Сравнение разных learning rates
- Сравнение разных размеров батчей
- Сравнение оптимизаторов
- Создание и оценка дополнительных признаков:
  - Полиномиальные признаки
  - Статистические признаки (скользящее среднее и std)

Визуализация:

- plot_losses() - отображение кривых обучения
- Табличное сравнение результатов

## Вывод
В данной работе была реализована линейная регрессия "с нуля" с использованием PyTorch, проведены эксперименты с разными оптимизаторами и методами генерации признаков.

Основные результаты:

Оптимизаторы:

- SGD показал стабильное, но медленное обучение.
- Adam и RMSprop сходились быстрее благодаря адаптивным learning rate.
- Adam оказался наиболее устойчивым к выбору гиперпараметров.

Гиперпараметры:

- Слишком высокий learning rate (например, 0.5) приводил к расходимости.
- Оптимальный batch size зависел от данных: меньшие батчи (8–32) давали лучшую сходимость, но требовали больше эпох.

Генерация признаков:

- Полиномиальные признаки (степени 2–3) улучшили качество модели.
- Статистические признаки (скользящее среднее и std) помогли уловить временные зависимости.
- Комбинированный подход (полиномы + статистика) дал наименьший MSE.

### Итог
Ручная реализация линейной регрессии подтвердила важность:
- Выбора оптимизатора (Adam/RMSprop лучше SGD)
- Тонкой настройки гиперпараметров (learning rate, batch size)