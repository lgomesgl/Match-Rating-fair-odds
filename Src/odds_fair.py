import numpy as np

def ajustar_percentuais(H, D, A):
    """
    Ajusta as porcentagens de H, D e A para que somem 100%.

    Parameters:
    H (float): Gols feitos pela equipe da casa.
    D (float): Gols empatados.
    A (float): Gols feitos pela equipe visitante.

    Returns:
    tuple: Porcentagens ajustadas de H, D e A.
    """
    # Garantir que os valores não sejam negativos
    H = max(H, 0)
    D = max(D, 0)
    A = max(A, 0)
    
    # Calcular a soma total
    total = H + D + A
    
    if total > 0:
        # Normalizar as porcentagens
        H_perc = (H / total) * 100
        D_perc = (D / total) * 100
        A_perc = (A / total) * 100
    else:
        # Se todos forem zero, atribuir 0% para cada um
        H_perc = D_perc = A_perc = 0
    
    return H_perc, D_perc, A_perc

def calcular_percentuais_ajustados(poly_h, poly_d, poly_a, x_min, x_max):
    """
    Calcula e ajusta as porcentagens de H, D e A para um intervalo de x.

    Parameters:
    poly_h (Polynomial): Polinômio para H.
    poly_d (Polynomial): Polinômio para D.
    poly_a (Polynomial): Polinômio para A.
    x_min (int): Valor mínimo de x.
    x_max (int): Valor máximo de x.

    Returns:
    dict: Dicionário contendo as porcentagens ajustadas para cada valor de x.
    """
    resultados = {}
    
    # Coeficientes e interceptos
    coefs_h = poly_h.coef_
    interc_h = poly_h.intercept_
    
    coefs_d = poly_d.coef_
    interc_d = poly_d.intercept_
    
    coefs_a = poly_a.coef_
    interc_a = poly_a.intercept_

    # Função de previsão
    def f(x, intercepto, coeficientes):
        return intercepto + sum(coeficientes[i] * (x ** i) for i in range(1, len(coeficientes)))

    # Para cada valor de x
    for x in range(x_min, x_max + 1):
        H_pred = f(x, interc_h, coefs_h)
        D_pred = f(x, interc_d, coefs_d)
        A_pred = f(x, interc_a, coefs_a)
        
        # Ajustar as porcentagens
        H_perc, D_perc, A_perc = ajustar_percentuais(H_pred, D_pred, A_pred)
        
        # Salvar no dicionário de resultados
        resultados[x] = {'H': round(H_perc, 2), 'D': round(D_perc, 2), 'A': round(A_perc, 2)}
    
    return resultados
