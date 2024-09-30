class MatchRating:
    def __init__(self, matchs_rating, estatistic):
        """
        Initializes the MatchRating class with the provided match ratings, statistic type, and league.
        
        :param matchs_rating: Dictionary to store match ratings.
        :param estatistic: The statistic to be used ('Gols', 'Shoots', 'Target Shoots').
        """
        self.estatistic = estatistic
        self.matchs_rating = matchs_rating
        
    def get_columns(self):
        """
        Maps the statistic type to the appropriate columns in the data.
        """
        columns_map = {
            'Gols': ['FTHG', 'FTAG'],
            'Shoots': ['HS', 'AS'],
            'Target Shoots': ['HST', 'AST']
        }
        
        self.columns = columns_map.get(self.estatistic)
        
    def get_match_rating(self, data, n_matchs_behind=5):
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
            
            def get_gols(team):
                """
                Calculates goals scored and conceded for a given team in the past matches.
                
                :param team: The team name for which to calculate goals.
                :return: Tuple of (goals scored, goals conceded).
                """
                feitos = 0
                concedidos = 0            
                
                # Goals for home matches
                data_home = data_behind[(data_behind['HomeTeam'] == team)]                
                feitos += int(data_home[self.columns[0]].sum())
                concedidos += int(data_home[self.columns[1]].sum())
                
                # Goals for away matches
                data_away = data_behind[(data_behind['AwayTeam'] == team)]
                feitos += int(data_away[self.columns[1]].sum())
                concedidos += int(data_away[self.columns[0]].sum())
                
                return feitos, concedidos
              
            # Calculate goals for home and away teams
            feitos_home, concedidos_home = get_gols(home_team)
            feitos_away, concedidos_away = get_gols(away_team)
            
            # Calculate match rating for both teams
            match_team_home = feitos_home - concedidos_home
            match_team_away = feitos_away - concedidos_away
            
            match_rating = match_team_home - match_team_away
             
            # Get the final result for the match
            ftr = row['FTR']
            
            # Update match ratings dictionary
            if match_rating not in self.matchs_rating[self.estatistic]:
                self.matchs_rating[self.estatistic][match_rating] = {'H': 0, 'D': 0, 'A': 0}
                
            # Update the corresponding outcome (H, D, A)
            if ftr in self.matchs_rating[self.estatistic][match_rating]:
                self.matchs_rating[self.estatistic][match_rating][ftr] += 1