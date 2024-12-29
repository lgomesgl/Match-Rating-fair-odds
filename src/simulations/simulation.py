import logging
import numpy as np
import pandas as pd
import os
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
from src.utils.tools import load_json_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MonteCarlo:
    def __init__(self, games: List[Dict[str, str | List]]):
        self.games = games

    @staticmethod
    def _simulate_batch(games: List[Dict[str, str | List]], number_of_samples: int) -> List[Dict]:
        """
        Simulate a single batch of games for a ticket randomly.
        """
        rng = np.random.default_rng()
        return rng.choice(games, size=number_of_samples, replace=False).tolist()

    def simulate(self, iterations: int, number_of_samples: int) -> List[List[Dict]]:
        """
        Simulate games for tickets randomly using parallelization.
        """
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                MonteCarlo._simulate_batch, 
                [self.games] * iterations, 
                [number_of_samples] * iterations
            ))
        return results

class Simulation:
    def __init__(self, matches):
        """
            Simulate matchs using monte carlo and combine them in a ticket
        """
        self.matchs = matches

    def extract_games_by_league(self, leagues_name: List[str] = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']):
        games = []
        count = 0
        for league, matches in self.matchs.items():
            if league in leagues_name:
                for match, probabilities in matches.items():
                    games.append({"league": league, 
                                  "match": match, 
                                  "probabilities": [probabilities["H"], probabilities["D"], probabilities["A"]]})
                    count += 1
        logging.info(f"Total games of process: { count }")      

        return games  
    
    def _restictions(self):
        pass
    
    def _get_probabilities(self, tickets: np.ndarray) -> np.ndarray:
        """
            Receive the tickets with the games and get the probabilities of each game.
            Apply the restrictions
        """
        rng = np.random.default_rng()
        results = ['Home', 'Draw', 'Away']

        tickets_probabilities = []
        for ticket in tickets:
            #print(ticket)
            ticket_probabilities = []
            for game in ticket:
                #print(game)
                prob_choice = rng.choice(results, p=game['probabilities'])

                ticket_probabilities.append({"league": game["league"], 
                                              "match": game["match"],
                                              "result": prob_choice})
            
            #if self._restictions():
            tickets_probabilities.append(ticket_probabilities)
        
        return tickets_probabilities


    def run():
        pass


if __name__ == '__main__':
    json_path = r"C:\home\projects\matchRating\database\json\simulation_probabilities.json"
    games = load_json_file(file_path=json_path)
    sl = Simulation(matches=games)
    g = sl.extract_games_by_league()
    mc = MonteCarlo(games=g)
    ga = mc.simulate(100, 5)
    ga_ = sl._get_probabilities(tickets=ga)