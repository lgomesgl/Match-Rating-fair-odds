import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def best_polynomial_fit(X, y, max_degree=4, threshold=0.05):
    best_degree = 1
    best_r2 = -np.inf
    best_model = None
    degrees = range(1, max_degree + 1)
    r2_values = []

    # Ajustar a regressão para cada grau de 1 a max_degree
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)

        r2_values.append(r2)
        # print(f"Modelo grau {degree}: R^2 = {r2:.4f}")

        # Verifica se o modelo melhora significativamente em relação ao anterior
        if r2 > best_r2 + threshold:
            best_r2 = r2
            best_degree = degree
            best_model = model

    return best_model, best_degree, best_r2

# Função para calcular as porcentagens e plotar os gráficos com regressão polinomial e linear
def plot_percentages_with_best_regression(data_dict):
    keys = np.array(list(data_dict.keys()))  # Usamos NumPy para calcular a regressão
    H_perc = []
    D_perc = []
    A_perc = []

    # Calcular porcentagens para H, D, A
    for key in keys:
        values = data_dict[key]
        total = sum(values.values())  # Soma dos valores de H, D, A
        if total > 0:
            H_perc.append((values['H'] / total) * 100)
            D_perc.append((values['D'] / total) * 100)
            A_perc.append((values['A'] / total) * 100)
        else:
            H_perc.append(0)
            D_perc.append(0)
            A_perc.append(0)

    # Converter listas para arrays do NumPy
    H_perc = np.array(H_perc)
    D_perc = np.array(D_perc)
    A_perc = np.array(A_perc)

    return keys, H_perc, D_perc, A_perc

def plot_regression(keys, data_perc, color, label, max_degree=4):
    plt.plot(keys, data_perc, f'{color}o', label=label)  # Original data points
    best_model, best_degree, best_r2 = best_polynomial_fit(keys, data_perc, max_degree=max_degree, threshold=0.02)
    
    poly = PolynomialFeatures(best_degree)
    keys_poly = poly.fit_transform(keys.reshape(-1, 1))
    predictions = best_model.predict(keys_poly)
    
    plt.plot(keys, predictions, f'{color}-', label=f'Regressão Polinomial {label} (grau {best_degree}): $R^2 = {best_r2:.2f}$')
    plt.title(f'Porcentagem de {label} - Melhor Ajuste $R^2$ = {best_r2:.2f}')
    plt.ylabel('Porcentagem (%)')
    plt.legend()
    plt.grid(True)

    return best_model, best_degree, best_r2
   