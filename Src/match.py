import os
import pandas as pd
import matplotlib.pyplot as plt

class MatchRating:
    def __init__(self, matchs_rating ,league='PL', estatistic='Gols'):
        self.league = league
        self.estatistic = estatistic
        self.matchs_rating = matchs_rating
        
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
                feitos += int(data_home['FTHG'].sum())
                concedidos += int(data_home['FTAG'].sum())
                
                data_away = data_behind[(data_behind['AwayTeam'] == team)]
                feitos += int(data_away['FTAG'].sum())
                concedidos += int(data_away['FTHG'].sum())
                
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
                self.matchs_rating[match_rating][ftr] += 1
                
# Função para calcular as porcentagens e plotar os gráficos
def plot_percentages(data_dict):
    keys = list(data_dict.keys())
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
                          # Criar gráficos
    plt.figure(figsize=(12, 8))

    # Gráfico de H
    plt.subplot(3, 1, 1)
    plt.plot(keys, H_perc, 'bo', label='H') 
    plt.title('Porcentagem de H')
    plt.ylabel('Porcentagem (%)')
    plt.grid(True)

    # Gráfico de D
    plt.subplot(3, 1, 2)
    plt.plot(keys, D_perc, 'go', label='D')
    plt.title('Porcentagem de D')
    plt.ylabel('Porcentagem (%)')
    plt.grid(True)

    # Gráfico de A
    plt.subplot(3, 1, 3)
    plt.plot(keys, A_perc, 'ro', label='A') 
    plt.title('Porcentagem de A')
    plt.ylabel('Porcentagem (%)')
    plt.grid(True)

    # Ajustar espaçamento entre os gráficos
    plt.tight_layout()
    plt.show()      
            
if __name__ == '__main__':
    file = r'D:\LUCAS\Match Rating\Database'
    
    datas = os.listdir(file)
    matchs_rating = {}
    for data in datas:
        df = pd.read_csv(os.path.join(file, data))
        test = MatchRating(matchs_rating=matchs_rating)
        test.get_match_rating(data=df)
    print(dict(sorted(matchs_rating.items())))
    
    plot_percentages(data_dict=dict(sorted(matchs_rating.items())))