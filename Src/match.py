import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from polynomial_fit import plot_percentages_with_best_regression, plot_regression
from odds_fair import calcular_percentuais_ajustados

class MatchRating:
    def __init__(self, matchs_rating, estatistic, league='PL'):
        self.league = league
        self.estatistic = estatistic
        self.matchs_rating = matchs_rating
        
    def get_columns(self):
        columns_map = {
            'Gols': ['FTHG', 'FTAG'],
            'Shoots': ['HS', 'AS'],
            'Target Shoots': ['HST', 'AST']
        }
        
        self.columns = columns_map.get(self.estatistic)
        
    def get_match_rating(self, data, n_matchs_behind=5):    
        for i in range(n_matchs_behind*10+1, data.shape[0]):
            data_behind = data.iloc[i-n_matchs_behind*10-1:i, :]

            row = data.loc[i]
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            def get_gols(team):
                feitos = 0
                concedidos = 0            
                    
                data_home = data_behind[(data_behind['HomeTeam'] == team)]                
                feitos += int(data_home[f'{ self.columns[0] }'].sum())
                concedidos += int(data_home[f'{ self.columns[1] }'].sum())
                
                data_away = data_behind[(data_behind['AwayTeam'] == team)]
                feitos += int(data_away[f'{ self.columns[1] }'].sum())
                concedidos += int(data_away[f'{ self.columns[0] }'].sum())
                
                return feitos, concedidos
              
            feitos_home, concedidos_home = get_gols(home_team)
            feitos_away, concedidos_away = get_gols(away_team)
            
            match_team_home = feitos_home-concedidos_home
            match_team_away = feitos_away-concedidos_away
            
            match_rating = match_team_home-match_team_away
             
            ftr = row['FTR']
            
            if match_rating not in self.matchs_rating:
                self.matchs_rating[match_rating] = {'H': 0, 'D':0, 'A':0}
                self.matchs_rating[match_rating][ftr] += 1
            else: 
                try:
                    self.matchs_rating[match_rating][ftr] += 1
                except:
                    pass                
                            
if __name__ == '__main__':
    file = r'D:\LUCAS\Match Rating\Database\Premier League'
    
    datas = os.listdir(file)
    matchs_rating = {}
    for data in datas:
        df = pd.read_csv(os.path.join(file, data))
        test = MatchRating(matchs_rating=matchs_rating, estatistic='Target Shoots')
        test.get_columns()
        test.get_match_rating(data=df)
    print(dict(sorted(matchs_rating.items())))
    
    keys, H_perc, D_perc, A_perc = plot_percentages_with_best_regression(data_dict=dict(sorted(matchs_rating.items())))
    
    plt.subplot(3, 1, 1)
    best_model_H, best_degree_H, best_r2_H = plot_regression(keys=keys, data_perc=H_perc, color='b', label='H')
    
    plt.subplot(3, 1, 2)
    best_model_D, best_degree_D, best_r2_D = plot_regression(keys=keys, data_perc=D_perc, color='g', label='D')

    plt.subplot(3, 1, 3)
    best_model_A, best_degree_A, best_r2_A = plot_regression(keys=keys, data_perc=A_perc, color='r', label='A')

    # plt.tight_layout()
    # plt.show()
    
    resultados = calcular_percentuais_ajustados(best_model_H, best_model_D, best_model_A, -40, 40)
    tabela = pd.DataFrame(resultados).T
    print(tabela)
    # # Exibir os resultados
    # for x, percentuais in resultados.items():
    #     print(f"x = {x}: H = {percentuais['H']:.2f}%, D = {percentuais['D']:.2f}%, A = {percentuais['A']:.2f}%")