import numpy as np
import os
import json
from typing import Union, Dict, Any

def normalize_data(data, method: str):
    """
        Normalizes the data based on the chosen method.
        
        :param data: Dictionary containing 'H', 'D', 'A' values for each key.
        :param method: The normalization method to apply. Options are ['min_max', 'z_score', 'direct', 'softmax', 'rank'].
        :return: Normalized data dictionary.
    """
    
    def min_max_normalize(values):
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val) / (max_val - min_val) for x in values]
    
    def z_score_normalize(values):
        mean = np.mean(values)
        std_dev = np.std(values)
        return [(x - mean) / std_dev for x in values]

    def direct_renormalize(values):
        total = np.sum(values)
        renormalize = 100 / total
        return [value * renormalize for value in values]
            
    def softmax(x):
        e_x = np.exp(x - np.max(x)) 
        return e_x / e_x.sum()

    def rank_normalization(values):
        # Get the ranks of the data
        ranks = np.argsort(np.argsort(values))
        # Normalize ranks to scale 0-1
        normalized = (ranks + 1) / (len(values) + 1)  # +1 to avoid zero
        return normalized
    
    # Extract H, D, A values from the data dictionary
    H_values = [d['H'] for d in data.values()]
    D_values = [d['D'] for d in data.values()]
    A_values = [d['A'] for d in data.values()]

    # Apply the chosen normalization method
    if method == 'min_max':
        H_values = min_max_normalize(H_values)
        D_values = min_max_normalize(D_values)
        A_values = min_max_normalize(A_values)

    elif method == 'z_score':
        H_values = z_score_normalize(H_values)
        D_values = z_score_normalize(D_values)
        A_values = z_score_normalize(A_values)

    elif method == 'direct':
        H_values = direct_renormalize(H_values)
        D_values = direct_renormalize(D_values)
        A_values = direct_renormalize(A_values)

    elif method == 'softmax':
        for i, key in enumerate(data.keys()):
            ratings = [H_values[i], D_values[i], A_values[i]]
            probabilities = softmax(ratings)
            H_values[i], D_values[i], A_values[i] = probabilities

    elif method == 'rank':
        H_values = rank_normalization(H_values)
        D_values = rank_normalization(D_values)
        A_values = rank_normalization(A_values)

    for i, key in enumerate(data.keys()):
        data[key]['H_normalized'] = H_values[i]
        data[key]['D_normalized'] = D_values[i]
        data[key]['A_normalized'] = A_values[i]
        
    return data

def initial_guess_value(league_name: str) -> float:
    league_values = {
        'Premier League': 0.45,
        'La Liga': 0.43,
        'Bundesliga': 0.63,
        'Serie A': 0.60,
        'Ligue 1': 0.55
    }

    if league_name not in league_values:
        raise ValueError(
            f"This league: '{league_name}' is not in the database. "
            "Please choose between 'Premier League', 'La Liga', 'Bundesliga', 'Serie A', or 'Ligue 1'."
        )

    return league_values[league_name]

def load_json_file(file_path: str) -> Union[Dict[str, Any], list, None]:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def save_json_file(output_path: str, json_data: Union[Dict[str, Any], list]) -> None:
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
