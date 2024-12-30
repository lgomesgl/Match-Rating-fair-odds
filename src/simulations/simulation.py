import logging
import numpy as np
import pandas as pd
import os
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
#from numba import njit
from src.utils.tools import load_json_file
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MonteCarlo:
    def __init__(self, games: List[Dict[str, str | List]]):
        self.games = games

    @staticmethod
    #@njit
    def _simulate_batch(games: np.ndarray, number_of_samples: int) -> np.ndarray:
        """
        Simulate a single batch of games for a ticket randomly using Numba.
        """
        rng = np.random.default_rng()
        return rng.choice(games, size=number_of_samples, replace=False)

    def simulate(self, iterations: int, number_of_samples: int) -> List[np.ndarray]:
        """
        Simulate games for tickets randomly using parallelization.
        """
        logging.info(f"Starting simulation with {iterations} iterations and {number_of_samples} samples per ticket.")
        start_time = time.time()

        games_ = np.array(self.games)
        with ProcessPoolExecutor() as executor:
            results = []
            for i, result in enumerate(executor.map(
                MonteCarlo._simulate_batch,
                [games_] * iterations,
                [number_of_samples] * iterations
            )):
                results.append(result)
                # Registra o progresso a cada 10% de iterações
                if (i + 1) % (iterations // 10) == 0:
                    progress = (i + 1) / iterations * 100
                    logging.info(f"{progress:.0f}% of tickets simulated")

        end_time = time.time()
        logging.info(f"Simulation completed in {end_time - start_time:.2f} seconds.")
        return results


class Simulation:
    def __init__(self, matches):
        self.matchs = matches
        self.best_tickets = []

    def extract_games_by_league(self, leagues_name: List[str] = None):
        if leagues_name is None:
            leagues_name = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
        
        games = []
        count = 0
        start_time = time.time()

        for league, matches in self.matchs.items():
            if league in leagues_name:
                for match, probabilities in matches.items():
                    games.append({"league": league,
                                  "match": match,
                                  "probabilities": [probabilities["H"], probabilities["D"], probabilities["A"]]})
                    count += 1

        end_time = time.time()
        logging.info(f"Extracted {count} games in {end_time - start_time:.2f} seconds.")
        return games

    @staticmethod
    def _restrictions(ticket: List[Dict[str, str]], restriction_league: bool) -> bool:
        league_count = Counter(game['league'] for game in ticket)
        result_count = Counter(game['result'] for game in ticket)
        ticket_size = len(ticket)

        if restriction_league and max(league_count.values(), default=0) > 2:
            return False
        if max(result_count.values(), default=0) == ticket_size:
            return False
        if result_count['Draw'] > 0.40 * ticket_size:
            return False
        if result_count['Away'] > 0.40 * ticket_size:
            return False

        return True

    def _get_probabilities(self, tickets: List[List[Dict]], restriction_league: bool = True) -> List[List[Dict]]:
        logging.info(f"Processing {len(tickets)} tickets for probabilities and restrictions.")
        start_time = time.time()

        rng = np.random.default_rng()
        results = ['Home', 'Draw', 'Away']
        tickets_probabilities = []

        for ticket in tickets:
            ticket_probabilities = []
            for game in ticket:
                prob_choice = rng.choice(results, p=game['probabilities'])
                ticket_probabilities.append({
                    "league": game["league"],
                    "match": game["match"],
                    "probabilities": game["probabilities"],
                    "result": prob_choice
                })

            if self._restrictions(ticket=ticket_probabilities, restriction_league=restriction_league):
                tickets_probabilities.append(ticket_probabilities)

        end_time = time.time()
        logging.info(f"Processed tickets in {end_time - start_time:.2f} seconds.")
        return tickets_probabilities

    def _get_best_tickets(self, tickets: List[List[Dict]], top_n: int = 5) -> List[List[Dict]]:
        logging.info(f"Calculating the best tickets from {len(tickets)} tickets.")
        start_time = time.time()

        scored_tickets = []
        for ticket in tickets:
            ticket_score = np.prod([
                game['probabilities'][['Home', 'Draw', 'Away'].index(game['result'])]
                for game in ticket
            ])
            scored_tickets.append((ticket, ticket_score))

        scored_tickets.sort(key=lambda x: x[1], reverse=True)
        self.best_tickets = [ticket for ticket, _ in scored_tickets[:top_n]]

        end_time = time.time()
        logging.info(f"Calculated the best tickets in {end_time - start_time:.2f} seconds.")
        return self.best_tickets

    def show_tickets(self):
        logging.info("Displaying the best tickets:")
        for i, ticket in enumerate(self.best_tickets, 1):
            print(f"Ticket {i}:")
            for game in ticket:
                print(f"  {game}")


if __name__ == '__main__':
    json_path = r"C:\home\projects\matchRating\database\json\simulation_probabilities.json"
    games = load_json_file(file_path=json_path)

    sl = Simulation(matches=games)
    extracted_games = sl.extract_games_by_league(leagues_name=['Premier League', 'Serie A'])
    mc = MonteCarlo(games=extracted_games)
    simulated_tickets = mc.simulate(iterations=100000, number_of_samples=5)
    processed_tickets = sl._get_probabilities(tickets=simulated_tickets, restriction_league=False)
    best_tickets = sl._get_best_tickets(tickets=processed_tickets)

    sl.show_tickets()
