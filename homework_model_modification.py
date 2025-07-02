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