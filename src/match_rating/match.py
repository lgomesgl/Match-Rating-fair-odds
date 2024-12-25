import pandas as pd
import os
from typing import Dict, Tuple, List, Optional
from weights.classification_table import LeagueTable
from utils.tools import load_json_file

root = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(root))

class MatchRating:
    def __init__(self, 
                 league_name: str,
                 matchs_rating: Dict, 
                 statistic: str, 
                 gols: float = 1.5):
        """
            Initializes the MatchRating class with the provided match ratings, statistic type, and league.
            
            Args:
                league_name (str): The name of the league
                matchs_rating (Dict): Dictionary to store match ratings.
                statistic (str): The statistic to be used ('Gols', 'Shoots', 'Target Shoots').
                gols (float): The threshold for goal classification (default is 1.5).
        """
        self.league_name = league_name
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

    def _get_gols(self, data: pd.DataFrame, team: str) -> Tuple[int, int]:
        """
            Calculates goals scored and conceded for a given team in the past matches.
            
            Args:
                data (pd.DataFrame): Data to search the goals of the team.
                team (str): The team name for which to calculate goals.

            Return: 
                Tuple of (goals scored, goals conceded).
        """
        score = 0
        conceded = 0            
        
        # Goals for home matches
        data_home = data[(data['HomeTeam'] == team)]        
        score += int(data_home[self.columns[0]].sum())
        conceded += int(data_home[self.columns[1]].sum())
        
        # Goals for away matches
        data_away = data[(data['AwayTeam'] == team)]
        score += int(data_away[self.columns[1]].sum())
        conceded += int(data_away[self.columns[0]].sum())
        
        return score, conceded
    
    def __get_table_classification(self, file_name: str) -> List[Dict[str, str | float | int]]:
        """
        Retrieve the league classification table from a JSON file.

        Args:
            file_name (str): The name of the file containing the league classification data.

        Returns:
            List[Dict[str, str | float | int]]: A list of dictionaries representing the league table data.
        """
        data = load_json_file(f'{ parent_path }/database/json/league_classifications.json')
        table = data[self.league_name][file_name]
        return table
    
    def __calculate_gols_classification(self,
                       score: float,
                       conceded: float,
                       table: Dict[int, Optional[pd.DataFrame]] | List[Dict[str, str | float | int]],
                       rows: pd.DataFrame, 
                       opponent_column: str, 
                       is_home: bool) -> Tuple[float, float]:
        """
        Calculate the weighted goals scored and conceded by a team based on its opponents' positions in the league table.

        Args:
            score (float): Initial score value (usually 0).
            conceded (float): Initial conceded value (usually 0).
            table (Dict[int, Optional[pd.DataFrame]] | List[Dict[str, str | float | int]]): 
            The league table or weights for classification.
            rows (pd.DataFrame): Match data rows for the team.
            opponent_column (str): Column name for the opposing team's identifier.
            is_home (bool): Indicates if the team is playing at home.

        Returns:
            Tuple[float, float]: Updated weighted goals scored and conceded.
        """
        weights_score_col = "weight score"
        weights_conceded_col = "weight conceded"

        for i, row in rows.iterrows():
            opponent_team = row[opponent_column]
            gols_scored = row['FTHG'] if is_home else row['FTAG']
            gols_conceded = row['FTAG'] if is_home else row['FTHG']

            if i == 0: 
                score += int(gols_scored)
                conceded += int(gols_conceded)                    
                continue

            table_at_row = table.get(i - 1)
            if table_at_row is None:  
                score += int(gols_scored)
                conceded += int(gols_conceded)            
                continue
            
            team_at_table = table_at_row[table_at_row['index'] == opponent_team]

            if team_at_table.empty:
                score += int(gols_scored)
                conceded += int(gols_conceded)   
                continue

            weight_score = float(team_at_table[weights_score_col].iloc[0])
            weight_conceded = float(team_at_table[weights_conceded_col].iloc[0])

            score += int(gols_scored) * weight_score
            conceded += int(gols_conceded) * weight_conceded

        return score, conceded

    def _get_gols_with_classification(
        self,
        file_name: str, 
        data: pd.DataFrame, 
        data_behind_n_matchs: pd.DataFrame, 
        team: str) -> Tuple[int, int]:
        """
        Calculate the total weighted goals scored and conceded by a team based on 
        the opponents' weights in the league table.

        Args:
            file_name (str): Name of the file containing league classification data.
            data (pd.DataFrame): Complete league data.
            data_behind_n_matchs (pd.DataFrame): Filtered data for the last N matches.
            team (str): Team name.

        Returns:
            Tuple[int, int]: Total weighted goals scored and conceded.
        """
        score = 0
        conceded = 0

        data_home = data_behind_n_matchs[data_behind_n_matchs['HomeTeam'] == team]
        data_away = data_behind_n_matchs[data_behind_n_matchs['AwayTeam'] == team]

        table = LeagueTable(data_league=data).fit()
        #table = self.__get_table_classification(file_name=file_name)
        
        score, conceded = self.__calculate_gols_classification(score=score, 
                                                               conceded=conceded, 
                                                               table=table, 
                                                               rows=data_home, 
                                                               opponent_column='AwayTeam', 
                                                               is_home=True)
        score, conceded = self.__calculate_gols_classification(score=score, 
                                                               conceded=conceded, 
                                                               table=table, 
                                                               rows=data_away, 
                                                               opponent_column='HomeTeam', 
                                                               is_home=False)
        
        return score, conceded

    def get_match_rating(self, 
                         file_name: str,
                         data: pd.DataFrame, 
                         n_matchs_behind: int = 5, 
                         classification: bool = False) -> None:
        """
            Calculates the match ratings based on the number of matches behind and updates the match ratings dictionary.
            
            Args:
                file_name (str): Name of files.
                data (pd.DataFrame): DataFrame containing the match data.
                n_matchs_behind (int): Number of matches to look back for calculating ratings (default is 5).
                classification (bool): Using the classification table to put weight (deafult False).
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
                score_home, conceded_home = self._get_gols(data=data_behind_n_matchs, team=home_team)
                score_away, conceded_away = self._get_gols(data=data_behind_n_matchs, team=away_team)
            else:
                # Calculate goals for home and away teams using weights of classification
                score_home, conceded_home = self._get_gols_with_classification(data=data,
                                                                               file_name=file_name, 
                                                                               data_behind_n_matchs=data_behind_n_matchs, 
                                                                               team=home_team)
                score_away, conceded_away = self._get_gols_with_classification(data=data, 
                                                                               file_name=file_name, 
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
                