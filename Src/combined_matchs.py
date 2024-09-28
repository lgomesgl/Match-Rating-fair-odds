import numpy as np
from optimizer import OptimizerAdam

class OneModel:
    def __init__(self, data, models_ratings):
        self.data = data
        self.models_ratings = models_ratings
        
    def __get_columns(self, stats):
        columns_map = {
            'Gols': ['FTHG', 'FTAG'],
            'Target Shoots': ['HST', 'AST']
        }
        
        self.columns = columns_map.get(stats)
       
    def __probability_match(self, w1, prob_gols, prob_ts):
        w2 = 1-w1
        prob_match = w1*np.array(prob_gols) + w2*np.array(prob_ts)
        return prob_match
    
    def __prob_match_real(self, ftr):
        if ftr == 'H':
            return [1,0,0] 
        elif ftr == 'D':
            return [0,1,0]
        elif ftr == 'A':
            return [0,0,1]

    def __erro_log_loss(self, prob_real, prob_match, epsilon=1e-10):
        prob_match = np.clip(prob_match, epsilon, 1-epsilon)
        erro = -np.sum(prob_real * np.log(prob_match))
        return erro
    
    def __derivative_erro_log_loss(self, prob_real, prob_match, prob_gols, prob_ts):    
        dev_erro = np.sum((np.array(prob_match) - np.array(prob_real)) * (np.array(prob_gols) - np.array(prob_ts)))
        return dev_erro
               
    def __stochastic_gradient(self, w1, dev_erro, learning_rate=0.00001):
        return w1 - learning_rate*dev_erro
                         
    def calculate_w1(self, w1, prob_gols, prob_ts, ftr, optimizer):
        prob_match = self.__probability_match(w1=w1, prob_gols=prob_gols, prob_ts=prob_ts)
        prob_real = self.__prob_match_real(ftr=ftr)
        
        erro_log = self.__erro_log_loss(prob_real=prob_real, prob_match=prob_match)
        dev_erro = self.__derivative_erro_log_loss(prob_real=prob_real, prob_match=prob_match, prob_gols=prob_gols, prob_ts=prob_ts)

        # w1 = self.__stochastic_gradient(w1, dev_erro=dev_erro)
        
        # Atualiza w1 usando o Adam optimizer
        w1 = optimizer.update(w1, dev_erro)
        
        return max(0, min(1, w1)) # att the w1
                     
    def get_match_rating(self, w1, optimizer, n_matchs_behind=5):
        data = self.data
        for i in range(n_matchs_behind*10+1, data.shape[0]):
            for stats in ['Gols', 'Target Shoots']:
                self.__get_columns(stats=stats)
                
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
                    self.prob_gols = list(self.models_ratings[stats][match_rating].values())
                elif stats ==  'Target Shoots':
                    self.prob_ts = list(self.models_ratings[stats][match_rating].values())
            
            w1 = self.calculate_w1(w1=w1, prob_gols=self.prob_gols, prob_ts=self.prob_ts, ftr=self.ftr, optimizer=optimizer)    
            # print(w1)
        return w1
