from typing import Dict, TypedDict, List, Optional
import pandas as pd

class TeamStats(TypedDict):
    points: int
    goals_score: int
    goals_conceded: int
    goals_diff: int

class LeagueTable:
    def __init__(self) -> None:
        """
        Class to represent and calculate a league table based on match results.
        
        Points system:
        - Win: +3 points
        - Draw: +1 point for both teams
        - Loss: 0 points

        Tie-breaking criteria:
        - Total goals difference (goals score - goals conceded) - European leagues.

        Attributes:
        - table (dict): Dictionary where keys are team names and values are their stats.
        """
        self.table: Dict[str, TeamStats] = {}

    def create_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs the league table from match data.

        Args:
            data (pd.DataFrame): DataFrame containing match results with the following columns:
                - 'HomeTeam' (str): Name of the home team.
                - 'AwayTeam' (str): Name of the away team.
                - 'FTHG' (int): Full-time home goals.
                - 'FTAG' (int): Full-time away goals.
                - 'FTR' (str): Full-time result ('H' for Home win, 'A' for Away win, 'D' for Draw).

        Returns:
            pd.DataFrame: DataFrame representing the sorted league table with columns:
                - 'index': Team name.
                - 'points': Total points of the team.
                - 'goals score': Total goals scored by the team.
                - 'goals conceded': Total goals conceded by the team.
                - 'goals diff': Goal difference (goals scored - goals conceded).
                - 'weight': Weight based on team position in the league.
        """

        if data.empty:
            raise ValueError(f"The data passed to { LeagueTable.__name__ } is empty")
        
        required_columns = {'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Data is missing required columns: {required_columns - set(data.columns)}")
        
        for _, row in data.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            home_goals = row['FTHG']
            away_goals = row['FTAG']
            result = row['FTR']

            # Ensure teams are in the table
            if home_team not in self.table:
                self.table[home_team] = {'points': 0, 'goals score': 0, 'goals conceded': 0, 'goals diff': 0}
            if away_team not in self.table:
                self.table[away_team] = {'points': 0, 'goals score': 0, 'goals conceded': 0, 'goals diff': 0}

            if result == 'H':
                # Home win
                self.table[home_team]['points'] += 3
                self.table[home_team]['goals score'] += home_goals
                self.table[home_team]['goals conceded'] += away_goals

                self.table[away_team]['goals score'] += away_goals
                self.table[away_team]['goals conceded'] += home_goals

            elif result == 'A':
                # Away win
                self.table[away_team]['points'] += 3
                self.table[away_team]['goals score'] += away_goals
                self.table[away_team]['goals conceded'] += home_goals

                self.table[home_team]['goals score'] += home_goals
                self.table[home_team]['goals conceded'] += away_goals

            elif result == 'D':
                # Draw
                self.table[home_team]['points'] += 1
                self.table[away_team]['points'] += 1
                self.table[home_team]['goals score'] += home_goals
                self.table[home_team]['goals conceded'] += away_goals

                self.table[away_team]['goals score'] += away_goals
                self.table[away_team]['goals conceded'] += home_goals

        df = pd.DataFrame(self.table).T
        df['goals diff'] = df['goals score'] - df['goals conceded']

        sorted_df = df.sort_values(by=['points', 'goals diff'], ascending=[False, False]).reset_index(drop=False)

        return sorted_df
    
    def create_weights(
        self, 
        data: pd.DataFrame, 
        weights: Optional[List[float]] = None, 
        num_sections: int = 3) -> pd.DataFrame:
        """
        Assign weights to teams based on their position in the table, divided into variable sections.

        Args:
        - data (pd.DataFrame): League table data.
        - weights (Optional[List[float]]): List of weights for each section. 
          If not provided, default weights are equally distributed.
        - num_sections (int): Number of sections to divide the table into. Default is 3.

        Returns:
        - pd.DataFrame: DataFrame with an additional 'weight' column.
        """
        if num_sections < 1:
            raise ValueError("The number of sections must be at least 1.")

        if weights is None:
            weights = [1.2 - (i / num_sections) for i in range(num_sections)]

        if len(weights) != num_sections:
            raise ValueError(f"Number of weights ({len(weights)}) must match the number of sections ({num_sections}).")

        n = data.shape[0]
        section_size = n // num_sections
        weight_score = []

        # Assign weights to 'weight score'
        for i in range(n):
            for section in range(num_sections):
                if i < (section + 1) * section_size or section == num_sections - 1:
                    weight_score.append(weights[section])
                    break

        # Create the inverse weights for 'weight conceded'
        weight_conceded = []
        for i in range(n):
            for section in range(num_sections):
                if i < (section + 1) * section_size or section == num_sections - 1:
                    weight_conceded.append(weights[num_sections - section - 1])  # Reverse section weights
                    break

        data['weight score'] = weight_score
        data['weight conceded'] = weight_conceded

        return data

