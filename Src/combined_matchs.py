import numpy as np

class OneModel:
    def __init__(self, data, models_ratings):
        self.data = data
        self.models_ratings = models_ratings
        
    def get_columns(self, stats):
        columns_map = {
            'Gols': ['FTHG', 'FTAG'],
            'Target Shoots': ['HST', 'AST']
        }
        
        self.columns = columns_map.get(stats)
       
    def probability_match(self, w1, probability_gols, probability_ts):
        w2 = 1-w1
        prob_match = w1*probability_gols + w2*probability_ts
        return prob_match
    
    def prob_match_real(self, ftr):
        if ftr == 'H':
            return [1,0,0] 
        elif ftr == 'D':
            return [0,1,0]
        elif ftr == 'A':
            return [0,0,1]

    def erro_log_loss(self, prob_real, prob_match_model):
        erro = -np.sum(prob_real * np.log(prob_match_model))
        return erro
    
    def derivative_erro_log_loss(self, prob_real, prob_match_model, prob_gols, prob_chute):
        dev_erro = np.sum((prob_match_model - prob_real) * (prob_gols - prob_chute))
        return dev_erro
               
    def stochastic_gradient(self, w1, dev_erro, learning_rate=0.001):
        return w1 - learning_rate*dev_erro
               
    def calculate_w1(self, prob_gols, prob_ts, ftr):
        ...
             
    def get_match_rating(self, data, stats, n_matchs_behind=5):
        for i in range(n_matchs_behind*10+1, data.shape[0]):
            for stats in ['Gols', 'Target Shoots']:
                self.get_columns(stats=stats)
                
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
                
                self.ftr = row['FTR']
            
                if stats == 'Gols':
                    self.probabilities_gols_match = list(self.models_ratings[stats][match_rating].values())
                elif stats ==  'Target Shoots':
                    self.probabilities_ts_match = list(self.models_ratings[stats][match_rating].values())
            
            
