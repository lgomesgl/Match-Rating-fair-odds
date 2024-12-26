import logging
import numpy as np
import pandas as pd
import os
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MonteCarlo:
    def __init__(self, games: List[Dict[str, str | List]]):
        self.games = games

    def _restrictions():
        pass
    
    def simulate():
        pass



class simulation:
    def __init__(self, matches):
        """
            Simulate matchs using monte carlo and combine them in a ticket
        """
        self.matchs = matches

    def extract_games_by_league(self, leagues_name: List[str]):
        games = []
        for league, matches in self.matchs.items():
            count = 0
            if league in leagues_name:
                for match, probabilities in matches:
                    games.append({"league": league, 
                                  "match": match, 
                                  "probabilities": [probabilities["H"], probabilities["D"], probabilities["A"]]})
                    count += 1
            logging.info(f"Total games of { league } process: { count }")        
    
