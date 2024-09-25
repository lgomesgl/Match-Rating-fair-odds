import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from tools import plot_and_table, normalize_data
from combined_matchs import OneModel

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
            
            if match_rating not in self.matchs_rating[self.estatistic]:
                self.matchs_rating[self.estatistic][match_rating] = {'H': 0, 'D':0, 'A':0}
                self.matchs_rating[self.estatistic][match_rating][ftr] += 1
            else: 
                try:
                    self.matchs_rating[self.estatistic][match_rating][ftr] += 1
                except:
                    pass                
                            
if __name__ == '__main__':
    # Train
    file_train = r'D:\LUCAS\Match Rating\Database\Premier League\train'
    datas_train = os.listdir(file_train)

    matchs_rating = {
        'Gols':{},
        'Target Shoots': {}
    }
    for data in datas_train:
        df = pd.read_csv(os.path.join(file_train, data))
        for stats in ['Gols','Target Shoots']:
            test = MatchRating(matchs_rating=matchs_rating, estatistic=stats)
            test.get_columns()
            test.get_match_rating(data=df)
            
    results_gols = plot_and_table(data_dict=dict(sorted(matchs_rating['Gols'].items())), range=(-28,28), show_grap=False)
    results_ts = plot_and_table(data_dict=dict(sorted(matchs_rating['Target Shoots'].items())), range=(-45,61), show_grap=False)
    
    all_results = {
        'Gols': results_gols,
        'Target Shoots': results_ts
    }
    
    # Test
    file_test = r'D:\LUCAS\Match Rating\Database\Premier League\test'
    datas_test = os.listdir(file_test)
    
    w1 = 0.5 # Chute inicial
    for data in datas_test:
        df = pd.read_csv(os.path.join(file_test, data))
        
        onemodel = OneModel(data=df, models_ratings=all_results)
        w1 = onemodel.get_match_rating(w1=w1)
    print(w1)
        
