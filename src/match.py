import pandas as pd
from typing import Dict, Tuple
from classification import LeagueTable

class MatchRating:
    def __init__(self, matchs_rating: Dict, statistic: str, gols: float = 1.5):
        """
            Initializes the MatchRating class with the provided match ratings, statistic type, and league.
            
            :param matchs_rating: Dictionary to store match ratings.
            :param statistic: The statistic to be used ('Gols', 'Shoots', 'Target Shoots').
            :param gols: The threshold for goal classification (default is 1.5).
        """
        self.matchs_rating = matchs_rating
        self.statistic = statistic
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
        
        if self.statistic not in columns_map:
            raise ValueError(f"This statistic: {self.statistic} its not to be use. Choose between 'Gols', 'Shoots', 'Target Shoots'")
        
        self.columns = columns_map[self.statistic]

    def _get_gols(self, data_behind_n_matchs: pd.DataFrame, team: str) -> Tuple[int, int]:
            """
                Calculates goals scored and conceded for a given team in the past matches.
                
                :param team: The team name for which to calculate goals.
                :return: Tuple of (goals scored, goals conceded).
            """
            score = 0
            conceded = 0            
            
            # Goals for home matches
            data_home = data_behind_n_matchs[(data_behind_n_matchs['HomeTeam'] == team)]        
            score += int(data_home[self.columns[0]].sum())
            conceded += int(data_home[self.columns[1]].sum())
            
            # Goals for away matches
            data_away = data_behind_n_matchs[(data_behind_n_matchs['AwayTeam'] == team)]
            score += int(data_away[self.columns[1]].sum())
            conceded += int(data_away[self.columns[0]].sum())
            
            return score, conceded
    
    def _get_gols_with_classification(self, data: pd.DataFrame, data_behind_n_matchs: pd.DataFrame, team: str) -> Tuple[int, int]:
        score = 0 
        conceded = 0

        data_home = data_behind_n_matchs[(data_behind_n_matchs['HomeTeam'] == team)]   
        data_away = data_behind_n_matchs[(data_behind_n_matchs['AwayTeam'] == team)]

        for i, row in data_home.iterrows():
            df = data.iloc[:i, :]
            away_team = row['AwayTeam']

            if i == 0:
                score += int(row['FTHG'])
                conceded += int(row['FTAG'])
                continue

            table = LeagueTable()
            sorted_table = table.create_table(data=df)            
            weights = table.create_weights(data=sorted_table, weights=[1.2, 1.0, 0.8])
            weight_data = weights[weights['index'] == away_team]

            if weight_data.empty:
                score += int(row['FTHG'])
                conceded += int(row['FTAG'])
                continue

            score += int(row['FTHG']) * float(weight_data['weight score'].iloc[0])
            conceded += int(row['FTAG']) * float(weight_data['weight conceded'].iloc[0])
          
        for i, row in data_away.iterrows():
            df = data.iloc[:i, :]
            home_team = row['HomeTeam']

            if i == 0:
                score += int(row['FTHG'])
                conceded += int(row['FTAG'])
                continue

            table = LeagueTable()
            sorted_table = table.create_table(data=df)            
            weights = table.create_weights(data=sorted_table, weights=[1.2, 1.0, 0.8])
            weight_data = weights[weights['index'] == home_team]
          
            if weight_data.empty:
                score += int(row['FTAG'])
                conceded += int(row['FTHG'])
                continue

            score += int(row['FTAG']) * float(weight_data['weight score'].iloc[0])
            conceded += int(row['FTHG']) * float(weight_data['weight conceded'].iloc[0])

        return score, conceded


    def get_match_rating(self, data: pd.DataFrame, n_matchs_behind:int = 5, classification: bool = False) -> None:
        """
            Calculates the match ratings based on the number of matches behind and updates the match ratings dictionary.
            
            :param data: DataFrame containing the match data.
            :param n_matchs_behind: Number of matches to look back for calculating ratings (default is 5).
        """
        # Iterate through matches, starting after the number of matches behind
        for i in range(n_matchs_behind*10+1, data.shape[0]):
            data_behind_n_matchs = data.iloc[i-n_matchs_behind*10-1:i, :]

            # Get home and away team from the current match row
            row = data.loc[i]
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            if not classification:
                # Calculate goals for home and away teams
                score_home, conceded_home = self._get_gols(data_behind_n_matchs=data_behind_n_matchs, team=home_team)
                score_away, conceded_away = self._get_gols(data_behind_n_matchs=data_behind_n_matchs, team=away_team)
            else:
                # Calculate goals for home and away teams
                score_home, conceded_home = self._get_gols_with_classification(data=data, 
                                                                               data_behind_n_matchs=data_behind_n_matchs, 
                                                                               team=home_team)
                score_away, conceded_away = self._get_gols_with_classification(data=data, 
                                                                               data_behind_n_matchs=data_behind_n_matchs, 
                                                                               team=away_team)

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
            if match_rating not in self.matchs_rating[self.statistic]:
                self.matchs_rating[self.statistic][match_rating] = {'H': 0, 'D': 0, 'A': 0, '+gols': 0, '-gols': 0}
                
            # Update the corresponding outcome (H, D, A)
            if ftr in self.matchs_rating[self.statistic][match_rating]:
                self.matchs_rating[self.statistic][match_rating][ftr] += 1
                
            # Update the corresponding outcome ('+gols', '-gols')
            if keys_gols in self.matchs_rating[self.statistic][match_rating]:
                self.matchs_rating[self.statistic][match_rating][keys_gols] += 1
                