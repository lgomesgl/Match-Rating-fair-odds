def ajustar_percentuais(H, D, A):
    # Garantir que os valores não sejam negativos
    H = max(H, 0)
    D = max(D, 0)
    A = max(A, 0)
    
    # Calcular a soma total
    total = H + D + A
    
    if total > 0:
        # Normalizar as porcentagens para que somem 100%
        H_perc = (H / total) * 100
        D_perc = (D / total) * 100
        A_perc = (A / total) * 100
    else:
        # Se todos forem zero, atribuir 0% para cada um
        H_perc = D_perc = A_perc = 0
    
    return H_perc, D_perc, A_perc

def calcular_percentuais_ajustados(poly_h, poly_d, poly_a, x_min, x_max):
    resultados = {}
    
    coefs_h = poly_h.coef_
    interc_h = poly_h.intercept_
    
    coefs_d = poly_d.coef_
    interc_d = poly_d.intercept_
    
    coefs_a = poly_a.coef_
    interc_a = poly_a.intercept_
    
    # Criar uma função de previsão usando os coeficientes
    def f(x, intercepto, coeficientes):
        resultado = intercepto  # Começa com o intercepto
        for i in range(1, len(coeficientes)):
            resultado += coeficientes[i] * (x ** i)  # Adiciona cada termo
        return resultado
        
    # Para cada valor de x (de x_min até x_max)
    for x in range(x_min, x_max + 1):
        # print(f"Calculando para x = {x}")
        
        H_pred = f(x, interc_h, coefs_h)
        D_pred = f(x, interc_d, coefs_d)
        A_pred = f(x, interc_a, coefs_a)
        
        # Ajustar as porcentagens
        # H_perc, D_perc, A_perc = ajustar_percentuais(H_pred, D_pred, A_pred)
        
        # Salvar no dicionário de resultados
        resultados[x] = {'H': round(H_pred,2), 'D': round(D_pred,2), 'A': round(A_pred,2)}
    
    return resultados