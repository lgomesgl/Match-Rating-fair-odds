import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from optimizer import OptimizerAdam

class OneModel:
    """
        A model to predict match outcomes based on historical data and ratings.

        Parameters:
        data (DataFrame): Historical match data.
        models_ratings (dict): Ratings models for different match statistics.
    """

    def __init__(self, data: pd.DataFrame, models_ratings: Dict[str, any]) -> None:
        self.data = data
        self.models_ratings = models_ratings

    def _get_columns(self, stats: str) -> None:
        """
            Maps the statistic type to the appropriate columns in the data.
        """
        columns_map = {
            'Gols': ['FTHG', 'FTAG'],
            'Target Shoots': ['HST', 'AST']
        }
        
        if stats not in columns_map:
            raise ValueError(f"This statistic: {stats} its not to be use. Choose between 'Gols', 'Target Shoots'")
        
        self.columns = columns_map[stats]

    def __probability_match(self, 
                            w1: float, 
                            prob_gols: List[float], 
                            prob_ts: List[float]) -> np.ndarray:
        """
        Calculate the weighted probability of a match outcome.

        Parameters:
        w1 (float): Weight for probabilities based on goals.
        prob_gols (List[float]): Probabilities based on goals.
        prob_ts (List[float]): Probabilities based on target shots.

        Returns:
        np.ndarray: Weighted probabilities.
        """
        w2 = 1 - w1
        return w1 * np.array(prob_gols) + w2 * np.array(prob_ts)

    def __prob_match_real(self, ftr: str) -> List[int]:
        """
        Convert final result to probability format.

        Parameters:
        ftr (str): Final result ('H' for home win, 'D' for draw, 'A' for away win).

        Returns:
        List[float]: One-hot encoded probabilities.
        """
        return [1, 0, 0] if ftr == 'H' else [0, 1, 0] if ftr == 'D' else [0, 0, 1]

    def __erro_log_loss(self, 
                        prob_real: List[float], 
                        prob_match: np.ndarray, 
                        epsilon: float = 1e-10) -> float:
        """
        Calculate log loss error.

        Parameters:
        prob_real (List[float]): True probabilities.
        prob_match (np.ndarray): Predicted probabilities.
        epsilon (float): Small constant to prevent division by zero.

        Returns:
        float: Log loss error.
        """
        prob_match = np.clip(prob_match, epsilon, 1 - epsilon)
        return -np.sum(prob_real * np.log(prob_match))

    def __derivative_erro_log_loss(self, 
                                   prob_real: List[float], 
                                   prob_match: np.ndarray, 
                                   prob_gols: List[float], 
                                   prob_ts: List[float]) -> float:    
        """
        Calculate the derivative of log loss error.

        Parameters:
        prob_real (List[float]): True probabilities.
        prob_match (np.ndarray): Predicted probabilities.
        prob_gols (List[float]): Probabilities based on goals.
        prob_ts (List[float]): Probabilities based on target shots.

        Returns:
        float: Gradient of the log loss error.
        """
        return np.sum((np.array(prob_match) - np.array(prob_real)) * (np.array(prob_gols) - np.array(prob_ts)))

    def calculate_w1(self, 
                     w1: float, 
                     prob_gols: List[float], 
                     prob_ts: List[float], 
                     ftr: str, 
                     optimizer) -> float:
        """
        Calculate the weight w1 using probabilities and the optimizer.

        Parameters:
        w1 (float): Initial weight.
        prob_gols (List[float]): Probabilities based on goals.
        prob_ts (List[float]): Probabilities based on target shots.
        ftr (str): Final result ('H', 'D', 'A').
        optimizer (Optimizer): Optimizer to update the weight.

        Returns:
        float: Updated weight w1.
        """
        prob_match = self.__probability_match(w1=w1, prob_gols=prob_gols, prob_ts=prob_ts)
        prob_real = self.__prob_match_real(ftr=ftr)

        #erro_log = self.__erro_log_loss(prob_real=prob_real, prob_match=prob_match)
        gradient_error = self.__derivative_erro_log_loss(prob_real=prob_real, 
                                                         prob_match=prob_match, 
                                                         prob_gols=prob_gols, 
                                                         prob_ts=prob_ts)

        # Update w1 using the Adam optimizer
        w1 = optimizer.update(w1, gradient_error)

        return max(0, min(1, w1))  # Keep w1 in the range [0, 1]

    def _get_gols(self, team: str, data_behind: pd.DataFrame) -> Tuple[int, int]:
        """
        Get goals scored and conceded by the team.

        Parameters:
        team (str): Team name.

        Returns:
        Tuple[int, int]: Goals scored (feitos) and conceded (concedidos).
        """
        feitos: int = 0
        concedidos: int = 0

        data_home = data_behind[data_behind['HomeTeam'] == team]
        feitos += int(data_home[self.columns[0]].sum())
        concedidos += int(data_home[self.columns[1]].sum())

        data_away = data_behind[data_behind['AwayTeam'] == team]
        feitos += int(data_away[self.columns[1]].sum())
        concedidos += int(data_away[self.columns[0]].sum())

        return feitos, concedidos
    
    def get_match_rating(self, 
                         w1: float, 
                         optimizer, 
                         n_matchs_behind: int = 5) -> float:
        """
        Get the match rating based on past performance and update w1.

        Parameters:
        w1 (float): Initial weight for goal probabilities.
        optimizer (Optimizer): Optimizer to update the weight.
        n_matchs_behind (int): Number of past matches to consider for analysis.

        Returns:
        float: Updated weight w1.
        """
        data = self.data
        for i in range(n_matchs_behind * 10 + 1, data.shape[0]):
            for stats in ['Gols', 'Target Shoots']:
                self._get_columns(stats=stats)
                data_behind = data.iloc[i - n_matchs_behind * 10 - 1:i, :]

                row = data.loc[i]
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']

                feitos_home, concedidos_home = self._get_gols(team=home_team, data_behind=data_behind)
                feitos_away, concedidos_away = self._get_gols(team=away_team, data_behind=data_behind)

                match_team_home = feitos_home - concedidos_home
                match_team_away = feitos_away - concedidos_away

                match_rating = match_team_home - match_team_away

                self.ftr = row['FTR']

                if stats == 'Gols':
                    self.prob_gols = list(self.models_ratings[stats][match_rating].values())[:3]
                elif stats == 'Target Shoots':
                    self.prob_ts = list(self.models_ratings[stats][match_rating].values())[:3]

            w1 = self.calculate_w1(w1=w1, 
                                   prob_gols=self.prob_gols, 
                                   prob_ts=self.prob_ts, 
                                   ftr=self.ftr, 
                                   optimizer=optimizer)
            # print(w1)
        return w1
