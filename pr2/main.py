import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.txt", sep='\t', encoding='Windows-1251')

df = df.apply(pd.to_numeric)

X = df.drop('Диагноз', axis=1).values
y = df['Диагноз'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")


import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Тепловая карта корреляций между признаками")
plt.show()


def cfs_score(features, target_corr, feature_corr_matrix):
    k = len(features)
    if k == 0:
        return 0

    mean_feature_target_corr = np.mean([target_corr[f] for f in features])

    sum_intercorr = 0
    for i, j in combinations(features, 2):
        sum_intercorr += abs(feature_corr_matrix.loc[i, j])

    mean_intercorr = sum_intercorr / (k * (k - 1) / 2) if k > 1 else 0

    score = (k * mean_feature_target_corr) / np.sqrt(k + k * (k - 1) * mean_intercorr)
    return score


target_corr = corr_matrix['Диагноз'].drop('Диагноз')
feature_corr_matrix = corr_matrix.drop('Диагноз', axis=0).drop('Диагноз', axis=1)

n_features = X.shape[1]
target_n = n_features - 2
best_subset = None
best_score = -np.inf

for features in combinations(corr_matrix.columns[:-1], target_n):
    features = list(features)
    current_score = cfs_score(features, target_corr, feature_corr_matrix)
    if current_score > best_score:
        best_score = current_score
        best_subset = features

print(f"Лучшее подмножество признаков по CFS: {best_subset}")

sorted_features = target_corr.abs().sort_values(ascending=True).index
naive_subset = sorted_features[2:].tolist()
print(f"Подмножество по наивному методу: {naive_subset}")


def prepare_data(X, feature_names, selected_features, scaler=None):
    col_indices = [list(feature_names).index(f) for f in selected_features]
    X_selected = X[:, col_indices]

    if scaler:
        X_selected = scaler.fit_transform(X_selected)

    X_selected = np.hstack([np.ones((X_selected.shape[0], 1)), X_selected])
    return X_selected


feature_names = df.columns[:-1]

X_train_cfs = prepare_data(X_train[:, 1:], feature_names, best_subset, StandardScaler())
X_test_cfs = prepare_data(X_test[:, 1:], feature_names, best_subset, StandardScaler())

beta_cfs = gradient_descent(X_train_cfs, y_train, alpha=0.01, epochs=10000)
y_pred_cfs = predict(X_test_cfs, beta_cfs)
accuracy_cfs = np.mean(y_pred_cfs == y_test)
print(f"Accuracy CFS model: {accuracy_cfs:.4f}")

X_train_naive = prepare_data(X_train[:, 1:], feature_names, naive_subset, StandardScaler())
X_test_naive = prepare_data(X_test[:, 1:], feature_names, naive_subset, StandardScaler())

beta_naive = gradient_descent(X_train_naive, y_train, alpha=0.01, epochs=10000)
y_pred_naive = predict(X_test_naive, beta_naive)
accuracy_naive = np.mean(y_pred_naive == y_test)
print(f"Accuracy Naive model: {accuracy_naive:.4f}")

print(f"Accuracy Full model: {accuracy:.4f}")