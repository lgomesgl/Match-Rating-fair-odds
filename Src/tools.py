import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from scipy.stats import norm

from polynomial_fit import plot_percentages_with_best_regression, plot_regression
from odds_fair import calcular_percentuais_ajustados

def plot_and_table(data_dict, range, show_grap=False):
    keys, H_perc, D_perc, A_perc = plot_percentages_with_best_regression(data_dict=data_dict)
    
    plt.subplot(3, 1, 1)
    best_model_H, best_degree_H, best_r2_H = plot_regression(keys=keys, data_perc=H_perc, color='b', label='H')
    
    plt.subplot(3, 1, 2)
    best_model_D, best_degree_D, best_r2_D = plot_regression(keys=keys, data_perc=D_perc, color='g', label='D')

    plt.subplot(3, 1, 3)
    best_model_A, best_degree_A, best_r2_A = plot_regression(keys=keys, data_perc=A_perc, color='r', label='A')

    if show_grap:
        plt.tight_layout()
        plt.show()
        
    resultados = calcular_percentuais_ajustados(best_model_H, best_model_D, best_model_A, range[0], range[1])
    # resultados = normalize_data(resultados)
    
    # tabela = pd.DataFrame(resultados).T
    
    return resultados

def normalize_data(data):
    def min_max_normalize(values):
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val) / (max_val - min_val) for x in values]
    
    def z_score_normalize(data):
        mean = np.mean(data)
        std_dev = np.std(data)
        return [(x - mean) / std_dev for x in data]

    def direct_renormalize(values):
        total = np.sum(values)
        renormalize = 100/total
        return[value*renormalize for value in values]
            
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # estabilidade numérica
        return e_x / e_x.sum()

    def rank_normalization(data):
        # Obter os ranks (ordens) dos dados
        ranks = np.argsort(np.argsort(data))
        # Normalizar os ranks para uma escala de 0 a 1
        normalized = (ranks + 1) / (len(data) + 1)  # +1 para evitar zero
        return normalized
    
    # Extraímos os valores de H, D, A
    H_values = [d['H'] for d in data.values()]
    D_values = [d['D'] for d in data.values()]
    A_values = [d['A'] for d in data.values()]

    # Normalização Min-Max
    # H_normalized = z_score_normalize(H_values)
    # D_normalized = z_score_normalize(D_values)
    # A_normalized = z_score_normalize(A_values)
    
    # Aplicamos Softmax para cada linha
    for i, key in enumerate(data.keys()):
        ratings = [H_values[i], D_values[i], A_values[i]]
        # probabilities = softmax(ratings)
        # probabilities = direct_renormalize(ratings)
        # ratings = rank_normalization(ratings)
        # Atualiza o dicionário original com as probabilidades ajustadas
        data[key]['H_zscore'] = ratings[0]
        data[key]['D_zscore'] = ratings[1]
        data[key]['A_zscore'] = ratings[2]
        
        # data[key]['H_z'] = norm.cdf(ratings[0])
        # data[key]['D_z'] = norm.cdf(ratings[1])
        # data[key]['A_z'] = norm.cdf(ratings[2])
        
    return data

def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def save_json_file(output_path, json_data):
    # Primeiro, converte todos os valores que não são compatíveis com JSON
    # json_data = convert_values_for_json(json_data)
    
    # Salva o arquivo JSON
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)