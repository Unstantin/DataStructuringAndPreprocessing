import dsv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


raw_data = dsv.read('[SubtitleTools.com] diabetes.txt')

# Создание DataFrame
columns = raw_data[0]
data = raw_data[1:]
df = pd.DataFrame(data, columns=columns)

# Преобразование данных в числовой формат
df = df.apply(pd.to_numeric)

# Разделение на признаки и целевую переменную
X = df.drop('Диагноз', axis=1).values
y = df['Диагноз'].values

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Добавление столбца единиц для intercept
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])


# Функция сигмоиды
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Градиентный спуск
def gradient_descent(X, y, alpha=0.01, epochs=1000):
    beta = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = X @ beta
        predictions = sigmoid(z)
        gradient = X.T @ (predictions - y)
        beta -= alpha * gradient
    return beta


# Обучение модели
beta = gradient_descent(X_train, y_train, alpha=0.01, epochs=10000)


# Предсказание на тестовой выборке
def predict(X, beta, threshold=0.5):
    return (sigmoid(X @ beta) >= threshold).astype(int)

y_pred = predict(X_test, beta)

# Оценка точности
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")