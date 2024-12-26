import numpy as np
import pandas as pd
import os
from itertools import combinations
from src.utils.tools import load_json_file
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

root = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(root))
dados = load_json_file(file_path=f'{ parent_path }/database/json/simulation_probabilities.json') 

# extract the game
games = []
for liga, partidas in dados.items():
    for partida, probs in partidas.items():
        games.append({"liga": liga, "partida": partida, "prob": [probs["H"], probs["D"], probs["A"]]})
logging.info(f"Total de games da Premier League processados: {len(games)}")

# Resultados possíveis
resultados_possiveis = ["Home", "Draw", "Away"]

# Função para gerar um bilhete com restrições
def gerar_bilhete_com_restricoes():
    while True:
        bilhete = []
        for jogo in games:
            resultado = np.random.choice(resultados_possiveis, p=jogo["prob"])
            bilhete.append(resultado)

        if bilhete.count("Home") == len(bilhete):
            logging.warning("Bilhete rejeitado: todas as partidas são 'Home'")
            continue

        if bilhete.count("Draw") > 0.4 * len(bilhete):
            logging.warning("Bilhete rejeitado: mais de 40% das partidas são 'Draw'")
            continue

        return bilhete

# Gerar bilhetes simulados
n_simulacoes = 10000
logging.info(f"Iniciando {n_simulacoes} simulações de bilhetes")
bilhetes = []
for i in range(n_simulacoes):
    if i % 1000 == 0:
        logging.info(f"Simulação {i}/{n_simulacoes}")

    bilhete = gerar_bilhete_com_restricoes()
    bilhetes.append(bilhete)

# Converter os bilhetes gerados em um DataFrame
logging.info("Convertendo bilhetes para DataFrame")
df_bilhetes = pd.DataFrame(bilhetes, columns=[j["partida"] for j in games])

# Criar combinações de jogos (exemplo: 5 jogos)
n_jogos_por_combinacao = 5
logging.info(f"Criando combinações de {n_jogos_por_combinacao} jogos")
comb_5_jogos = list(combinations(range(len(games)), n_jogos_por_combinacao))

# Calcular as probabilidades combinadas
logging.info("Calculando probabilidades combinadas corretamente")
probabilidades_combinadas = []
for comb in comb_5_jogos:
    prob_comb = 1
    resultados_comb = []
    for i in comb:
        jogo = games[i]
        prob = jogo["prob"]
        # Resultado do bilhete para este jogo
        resultado_escolhido = df_bilhetes.iloc[0, i]
        # Probabilidade do resultado escolhido
        if resultado_escolhido == "Home":
            prob_comb *= prob[0]
        elif resultado_escolhido == "Draw":
            prob_comb *= prob[1]
        elif resultado_escolhido == "Away":
            prob_comb *= prob[2]
        resultados_comb.append(resultado_escolhido)
    probabilidades_combinadas.append((comb, prob_comb, resultados_comb))

# Ordenar combinações por probabilidade total (descendente)
logging.info("Ordenando combinações por probabilidade total")
probabilidades_combinadas.sort(key=lambda x: x[1], reverse=True)

# Selecionar as 5 melhores combinações
logging.info("Selecionando as 5 melhores combinações")
melhores_combinacoes = probabilidades_combinadas[:5]

# Exibir as 5 melhores combinações
logging.info("Exibindo as 5 melhores combinações")
for comb, prob, resultados in melhores_combinacoes:
    jogos_comb = [games[i]["partida"] for i in comb]
    resultados_str = [f"{jogo}: {res}" for jogo, res in zip(jogos_comb, resultados)]
    print(f"Combinação: {resultados_str} - Probabilidade: {prob:.4f}")