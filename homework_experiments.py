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

# Задание 3.2

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

