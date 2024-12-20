from typing import Dict, Tuple
import pandas as pd

class MatchRating:
    def __init__(self, matchs_rating: Dict, estatistic: str, gols: float = 1.5):
        """
            Initializes the MatchRating class with the provided match ratings, statistic type, and league.
            
            :param matchs_rating: Dictionary to store match ratings.
            :param estatistic: The statistic to be used ('Gols', 'Shoots', 'Target Shoots').
        """
        self.matchs_rating = matchs_rating
        self.estatistic = estatistic
        self.gols = gols
        
    def get_columns(self) -> None:
        """
            Maps the statistic type to the appropriate columns in the data.
        """
        columns_map = {
            'Gols': ['FTHG', 'FTAG'],
            'Shoots': ['HS', 'AS'],
            'Target Shoots': ['HST', 'AST']
        }
        
        self.columns = columns_map.get(self.estatistic)

        if not self.columns:
            raise ValueError(f"This statistic its not to be use. Choose between 'Gols', 'Shoots', 'Target Shoots'")
        
    def _get_gols(self, data_behind: pd.DataFrame, team: str) -> Tuple[int, int]:
            """
                Calculates goals scored and conceded for a given team in the past matches.
                
                :param team: The team name for which to calculate goals.
                :return: Tuple of (goals scored, goals conceded).
            """
            score = 0
            conceded = 0            
            
            # Goals for home matches
            data_home = data_behind[(data_behind['HomeTeam'] == team)]                
            score += int(data_home[self.columns[0]].sum())
            conceded += int(data_home[self.columns[1]].sum())
            
            # Goals for away matches
            data_away = data_behind[(data_behind['AwayTeam'] == team)]
            score += int(data_away[self.columns[1]].sum())
            conceded += int(data_away[self.columns[0]].sum())
            
            return score, conceded

    def get_match_rating(self, data: pd.DataFrame, n_matchs_behind:int = 5) -> None:
        """
            Calculates the match ratings based on the number of matches behind and updates the match ratings dictionary.
            
            :param data: DataFrame containing the match data.
            :param n_matchs_behind: Number of matches to look back for calculating ratings (default is 5).
        """
        # Iterate through matches, starting after the number of matches behind
        for i in range(n_matchs_behind*10+1, data.shape[0]):
            data_behind = data.iloc[i-n_matchs_behind*10-1:i, :]

            # Get home and away team from the current match row
            row = data.loc[i]
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            # Calculate goals for home and away teams
            score_home, conceded_home = self._get_gols(data_behind=data_behind, team=home_team)
            score_away, conceded_away = self._get_gols(data_behind=data_behind, team=away_team)
            
            # Calculate match rating for both teams
            match_team_home = score_home - conceded_home
            match_team_away = score_away - conceded_away
            
            match_rating = match_team_home - match_team_away
             
            # Get the final result for the match
            ftr = row['FTR']
            
            # Gols in the match
            gols_match = row['FTHG'] + row['FTAG']      
            if gols_match > self.gols:
                keys_gols = '+gols'
            else:
                keys_gols = '-gols'
            
            # Update match ratings dictionary
            if match_rating not in self.matchs_rating[self.estatistic]:
                self.matchs_rating[self.estatistic][match_rating] = {'H': 0, 'D': 0, 'A': 0, '+gols': 0, '-gols': 0}
                
            # Update the corresponding outcome (H, D, A)
            if ftr in self.matchs_rating[self.estatistic][match_rating]:
                self.matchs_rating[self.estatistic][match_rating][ftr] += 1
                
            # Update the corresponding outcome ('+gols', '-gols')
            if keys_gols in self.matchs_rating[self.estatistic][match_rating]:
                self.matchs_rating[self.estatistic][match_rating][keys_gols] += 1
                