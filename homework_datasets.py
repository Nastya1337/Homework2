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
    house_prices_path = './house_prices.csv'
    customer_churn_path = './customer_churn.csv'

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