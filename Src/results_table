import pandas as pd


class Table:

    def __init__(self, league):
        self.league = league
        self.table = {}

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
    

    
