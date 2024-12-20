import numpy as np
from optimizer import OptimizerAdam

class OneModel:
    """
        A model to predict match outcomes based on historical data and ratings.

        Parameters:
        data (DataFrame): Historical match data.
        models_ratings (dict): Ratings models for different match statistics.
    """

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
        w2 = 1 - w1
        return w1 * np.array(prob_gols) + w2 * np.array(prob_ts)

    def __prob_match_real(self, ftr):
        """Convert final result to probability format."""
        return [1, 0, 0] if ftr == 'H' else [0, 1, 0] if ftr == 'D' else [0, 0, 1]

    def __erro_log_loss(self, prob_real, prob_match, epsilon=1e-10):
        """Calculate log loss error."""
        prob_match = np.clip(prob_match, epsilon, 1 - epsilon)
        return -np.sum(prob_real * np.log(prob_match))

    def __derivative_erro_log_loss(self, prob_real, prob_match, prob_gols, prob_ts):    
        """Calculate the derivative of log loss error."""
        return np.sum((np.array(prob_match) - np.array(prob_real)) * (np.array(prob_gols) - np.array(prob_ts)))

    def calculate_w1(self, w1, prob_gols, prob_ts, ftr, optimizer):
        """Calculate the weight w1 using probabilities and the optimizer."""
        prob_match = self.__probability_match(w1=w1, prob_gols=prob_gols, prob_ts=prob_ts)
        prob_real = self.__prob_match_real(ftr=ftr)

        erro_log = self.__erro_log_loss(prob_real=prob_real, prob_match=prob_match)
        gradient_error = self.__derivative_erro_log_loss(prob_real=prob_real, prob_match=prob_match, prob_gols=prob_gols, prob_ts=prob_ts)

        # Update w1 using the Adam optimizer
        w1 = optimizer.update(w1, gradient_error)

        return max(0, min(1, w1))  # Keep w1 in the range [0, 1]

    def get_match_rating(self, w1, optimizer, n_matchs_behind=5):
        """Get the match rating based on past performance and update w1."""
        data = self.data
        for i in range(n_matchs_behind * 10 + 1, data.shape[0]):
            for stats in ['Gols', 'Target Shoots']:
                self.__get_columns(stats=stats)
                data_behind = data.iloc[i - n_matchs_behind * 10 - 1:i, :]

                row = data.loc[i]
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']

                def get_gols(team):
                    """Get goals scored and conceded by the team."""
                    feitos = 0
                    concedidos = 0

                    data_home = data_behind[data_behind['HomeTeam'] == team]
                    feitos += int(data_home[self.columns[0]].sum())
                    concedidos += int(data_home[self.columns[1]].sum())

                    data_away = data_behind[data_behind['AwayTeam'] == team]
                    feitos += int(data_away[self.columns[1]].sum())
                    concedidos += int(data_away[self.columns[0]].sum())

                    return feitos, concedidos
                
                feitos_home, concedidos_home = get_gols(home_team)
                feitos_away, concedidos_away = get_gols(away_team)

                match_team_home = feitos_home - concedidos_home
                match_team_away = feitos_away - concedidos_away

                match_rating = match_team_home - match_team_away

                self.ftr = row['FTR']

                if stats == 'Gols':
                    self.prob_gols = list(self.models_ratings[stats][match_rating].values())[:3]
                elif stats == 'Target Shoots':
                    self.prob_ts = list(self.models_ratings[stats][match_rating].values())[:3]

            w1 = self.calculate_w1(w1=w1, prob_gols=self.prob_gols, prob_ts=self.prob_ts, ftr=self.ftr, optimizer=optimizer)
            # print(w1)
        return w1
