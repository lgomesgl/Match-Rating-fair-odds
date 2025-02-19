import logging
import numpy as np
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from tqdm import tqdm  
from src.utils.tools import load_json_file
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MonteCarlo:
    def __init__(self, games: List[Dict[str, str | List]]):
        self.games = np.array(games)

    @staticmethod
    def _simulate_batch(games: np.ndarray, number_of_games: int, rng_seed: int) -> np.ndarray:
        """
        Simulate a single batch of games for a ticket randomly.
        Each process gets its own RNG derived from a unique seed.

        Args:
            games (np.ndarray): The games "in the box" to choose from.
            number_of_games (int): The number of games to select.
            rng_seed (int): The seed to use in random generation.

        Returns:
            np.ndarray: An array containing the randomly selected games.
        """
        rng = np.random.default_rng(rng_seed) 
        return rng.choice(games, size=number_of_games, replace=False)

    def simulate(self, iterations: int, number_of_games: int) -> List[np.ndarray]:
        """
        Simulates games for tickets randomly using parallelization.
        Ensures independent random number generation across parallel processes.

        Args:
            iterations (int): The number of simulations to run.
            number_of_games (int): The number of games to select in each simulation.

        Returns:
            List[np.ndarray]: A list of arrays, where each array represents a ticket with the selected games.
        """
        logging.info(f"Starting simulation with {iterations} iterations and {number_of_games} samples per ticket.")
        start_time = time.time()

        games_ = np.array(self.games)

        unique_seed = np.random.SeedSequence().entropy  
        seed_sequence = np.random.SeedSequence(int(unique_seed))
        rng_seeds = seed_sequence.spawn(iterations)  # Create independent seeds

        # Run parallel simulations
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(
                    MonteCarlo._simulate_batch,
                    [games_] * iterations,        
                    [number_of_games] * iterations, 
                    rng_seeds                       
                ),
                total=iterations, desc="Simulating games"
            ))

        end_time = time.time()
        logging.info(f"Simulation completed in {end_time - start_time:.2f} seconds.")
        return results

class Simulation:
    def __init__(self, matches):
        self.matchs = matches
        self.best_tickets = []

    def extract_games_by_league(self, leagues_name: List[str] = None):
        """
        Extracts games from the provided match data filtered by the specified leagues.

        Args:
            leagues_name (List[str], optional): A list of league names to filter games. 
                                                Defaults to common European leagues.

        Returns:
            List[dict]: A list of dictionaries, each representing a game with its league, match, 
                        and probabilities for 'Home', 'Draw', and 'Away'.
        """
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
    def _apply_restrictions(ticket: List[Dict[str, str]], 
                      restriction_league: bool, 
                      restriction_match: bool) -> bool:
        """
        Applies restrictions to a ticket to ensure it meets the specified conditions.

        Args:
            ticket (List[Dict[str, str]]): A list of games in the ticket.
            restriction_league (bool): Whether to restrict the maximum games per league.
            restriction_match (bool): Whether to restrict the ticket to avoid identical results.

        Returns:
            bool: True if the ticket meets the restrictions, False otherwise.
        """
        league_count = Counter(game['league'] for game in ticket)
        result_count = Counter(game['result'] for game in ticket)
        ticket_size = len(ticket)

        if restriction_league and max(league_count.values(), default=0) > 2:
            return False
        if restriction_match and max(result_count.values(), default=0) == ticket_size:
            return False
        if result_count['Draw'] > 0.40 * ticket_size:
            return False
        if result_count['Away'] > 0.40 * ticket_size:
            return False
    
        return True
    
    @staticmethod
    def _remove_duplicate_tickets(tickets: List[List[Dict]]) -> List[List[Dict]]:
        """
        Removes duplicate tickets by comparing their league, match, and result.

        Args:
            tickets (List[List[Dict]]): A list of tickets, where each ticket is a list of games.

        Returns:
            List[List[Dict]]: A list of unique tickets.
        """
        logging.info(f"Removing duplicates from {len(tickets)} tickets.")
        start_time = time.time()

        unique_tickets = set()
        deduplicated_tickets = []

        for ticket in tickets:
            ticket_representation = tuple(
                (game['league'], game['match'], game['result'])
                for game in sorted(ticket, key=lambda x: (x['league'], x['match']))
            )
            if ticket_representation not in unique_tickets:
                unique_tickets.add(ticket_representation)
                deduplicated_tickets.append(ticket)

        end_time = time.time()
        logging.info(f"Removed duplicates in {end_time - start_time:.2f} seconds. {len(deduplicated_tickets)} unique tickets remain.")
        return deduplicated_tickets

    def _validate_tickets(self, tickets: List[List[Dict]], 
                           restriction_league: bool = True, 
                           restriction_match: bool = True) -> List[List[Dict]]:
        """
        Validates tickets by applying probabilities and restrictions, and removes duplicates.

        Args:
            tickets (List[List[Dict]]): A list of tickets to validate.
            restriction_league (bool, optional): Whether to restrict the maximum games per league. Defaults to True.
            restriction_match (bool, optional): Whether to restrict identical results in a ticket. Defaults to True.

        Returns:
            List[List[Dict]]: A list of valid and unique tickets.
        """
        logging.info(f"Processing {len(tickets)} tickets for probabilities and restrictions.")
        start_time = time.time()

        rng = np.random.default_rng()
        results = ['Home', 'Draw', 'Away']
        processed_tickets = []

        for ticket in tickets:
            ticket_probabilities = [
                {**game, "result": rng.choice(results, p=game['probabilities'])} for game in ticket
            ]

            if self._apply_restrictions(ticket=ticket_probabilities, restriction_league=restriction_league, restriction_match=restriction_match):
                processed_tickets.append(ticket_probabilities)

        unique_tickets = self._remove_duplicate_tickets(processed_tickets)
        logging.info(f"Processed tickets in {time.time() - start_time:.2f} seconds.")
        return unique_tickets

    def _get_best_tickets(self, tickets: List[List[Dict]], top_n: int = 5) -> List[List[Dict]]:
        """
        Calculates the best tickets based on the product of probabilities.

        Args:
            tickets (List[List[Dict]]): A list of validated tickets.
            top_n (int, optional): The number of top tickets to return. Defaults to 5.

        Returns:
            List[List[Dict]]: A list of the top tickets.
        """
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
        """
        Displays the best tickets in the console, formatted for readability.
        """
        logging.info("Displaying the best tickets:")
        for i, ticket in enumerate(self.best_tickets, 1):
            print(f"Ticket {i}:")
            for game in ticket:
                print(f"  {game}")

    def run(self, 
            leagues_name, 
            iterations, 
            number_of_games, 
            restriction_league=False, 
            restriction_match=False):
        """
        Runs the entire simulation pipeline: extraction, simulation, validation, and ranking.

        Args:
            leagues_name (List[str]): A list of league names to filter games.
            iterations (int): The number of simulations to run.
            number_of_games (int): The number of games to include in each simulated ticket.
            restriction_league (bool, optional): Whether to restrict the maximum games per league. Defaults to False.
            restriction_match (bool, optional): Whether to restrict identical results in a ticket. Defaults to False.
        """
        extracted_games = self.extract_games_by_league(leagues_name=leagues_name)
        monte_carlo = MonteCarlo(games=extracted_games)
        simulated_tickets = monte_carlo.simulate(iterations=iterations, number_of_games=number_of_games)
        processed_tickets = self._validate_tickets(tickets=simulated_tickets, restriction_league=restriction_league, restriction_match=restriction_match)
        self._get_best_tickets(tickets=processed_tickets)
        self.show_tickets()
        

if __name__ == '__main__':
    json_path = r"C:\home\projects\matchRating\database\json\simulation_probabilities.json"
    games = load_json_file(file_path=json_path)

    simulation = Simulation(matches=games)
    simulation.run(leagues_name=None, 
                   iterations=300000, 
                   number_of_games=5, 
                   restriction_league=True, 
                   restriction_match=False)
