import pandas as pd


class Table:

    def __init__(self, league):
        """ 
        table class sorted by points in each match
        each table class consists of a table dictionary in which teams are the keys and the values are they're current 
        points in the league, table league consists of the league name which is being analyzed
        
        win : +3 points
        draw: +1 points for both teams
        loss: no points

        in case two teams are tied in points tie breaker parameters are decided in the following order:

        goals > goals taken        
        """
        self.league = league
        self.table = {}

    #create the table based on .csv given
    def table(self, path):
        dados = pd.read_csv(path)
        for i in range(1, len(dados)):
            line =dados.iloc[i]
            if line['FTR'] != 'D':
                result = 'HomeTeam'
                if line['FTR'] == 'A':
                    result = 'AwayTeam'
                if line[result] not in self.table:
                    line[result] = 0
                line[result] += 3
            else:
                if line['HomeTeam'] not in self.table:
                    line['HomeTeam'] = 0
                if line['AwayTeam'] not in self.table:
                    line['AwayTeam'] = 0
                line['HomeTeam'] += 1
                line['AwayTeam'] += 1
        self.table = sorted(self.table.items())
    

    
