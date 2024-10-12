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

        in case two teams are tied in points tie breaker parameters are decided with goals      
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
                letter = 'FTHG'
                if line['FTR'] == 'A':
                    result = 'AwayTeam'
                    letter = 'FTAG'
                if line[result] not in self.table:
                    self.table[line[result]] = {'points' : 0, 'goals' : 0, 'w': 0}
                self.table[line[result]][line[result]]['points'] += 3
                self.table[line[result]][line[result]]['goals'] += line[letter]
                
            else:
                if line['HomeTeam'] not in self.table:
                    self.table[line['HomeTeam']] = {'points': 0, 'goals' : 0, 'w' : 0}
                if line['AwayTeam'] not in self.table:
                    self.table[line['AwayTeam']] = {'points': 0, 'goals' : 0, 'w' : 0}
                self.table[line['HomeTeam']]['points'] += 1
                self.table[line['AwayTeam']]['points'] += 1
                self.table[line['HomeTeam']]['goals'] += line['FTHG']
                self.table[line['AwayTeam']]['goals'] += line['FTAG']

        self.table = sorted(self.table.items(), key = lambda x : (x['points'], x['goals']))
        n = len(self.table)
        for i, (chave, valor) in enumerate(self.table):
            if i < n // 3:
                valor['w'] = 0.4
            elif i < 2 * n // 3:
                valor['w'] = 0.6
            else:
                valor ['w'] = 0.8
        

    

    
