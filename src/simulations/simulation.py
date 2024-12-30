import logging
import numpy as np
import pandas as pd
import os
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
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
        self.best_tickets = []

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
    
    @staticmethod
    def _restrictions(ticket: List[Dict[str, str]], restriction_league: bool) -> bool:
        """
        Apply restrictions to the ticket.
        """
        league_count = Counter(game['league'] for game in ticket)
        result_count = Counter(game['result'] for game in ticket)
        ticket_size = len(ticket)

        # Restriction 1: No league should have more than 2 games
        if restriction_league:
            if max(league_count.values(), default=0) > 2:
                return False
        
        # Restriction 2: No result should be only in a ticket
        if max(result_count.values(), default=0) == ticket_size:
            return False
        
        # Restriction 3: No more them 40% draw games in a ticket
        if result_count['Draw'] > 0.40 * ticket_size:
            return False
        
        # Restriction 4: No more them 40% away games in a ticket
        if result_count['Away'] > 0.40 * ticket_size:
            return False

        return True

    def _get_probabilities(self, tickets: np.ndarray, restriction_league: bool = True) -> np.ndarray:
        """
            Receive the tickets with the games and get the probabilities of each game.
            Apply the restrictions
        """
        rng = np.random.default_rng()
        results = ['Home', 'Draw', 'Away']

        tickets_probabilities = []
        for ticket in tickets:
            ticket_probabilities = []
            for game in ticket:
                prob_choice = rng.choice(results, p=game['probabilities'])

                ticket_probabilities.append({"league": game["league"], 
                                              "match": game["match"],
                                              "probabilities": game["probabilities"],
                                              "result": prob_choice})
            
            if self._restrictions(ticket=ticket_probabilities, restriction_league=restriction_league):
                tickets_probabilities.append(ticket_probabilities)
        
        return tickets_probabilities

    def _get_best_tickets(self, tickets: List[List[Dict]], top_n: int = 5) -> List[List[Dict]]:
        """
        Select the best tickets based on the product of probabilities for the chosen results.

        Args:
            tickets: A list of tickets where each ticket is a list of game dictionaries.
            top_n: The number of top tickets to return.

        Returns:
            A list of the top N tickets based on their total probabilities.
        """
        scored_tickets = []

        for ticket in tickets:
            ticket_score = 1.0  
            for game in ticket:
                chosen_prob = game['probabilities'][['Home', 'Draw', 'Away'].index(game['result'])]
                ticket_score *= chosen_prob

            scored_tickets.append((ticket, ticket_score))

        scored_tickets.sort(key=lambda x: x[1], reverse=True)
        self.best_tickets = [ticket for ticket, _ in scored_tickets[:top_n]]

        return self.best_tickets

    def show_tickets(self):
        print("Best Tickets:")
        for i, ticket in enumerate(self.best_tickets, 1):
            print(f"Ticket {i}:")
            for game in ticket:
                print(f"  {game}")
        
    def run():
        pass


if __name__ == '__main__':
    json_path = r"C:\home\projects\matchRating\database\json\simulation_probabilities.json"
    games = load_json_file(file_path=json_path)
    sl = Simulation(matches=games)
    g = sl.extract_games_by_league()
    mc = MonteCarlo(games=g)
    ga = mc.simulate(100000, 5)
    ga_ = sl._get_probabilities(tickets=ga, restriction_league=True)
    best = sl._get_best_tickets(tickets=ga_)
    sl.best_tickets()