import os
import json
from src.utils.tools import load_json_file, save_json_file

root = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(root))

data_prob = load_json_file(file_path=f'{ parent_path }/database/json/fair_odds.json')

def normalize_probabilities(data):
    normalized_data = {}
    for league, matches in data.items():
        normalized_data[league] = {}
        count = 0
        for match, stats in matches.items():
            h = stats["H(%)"] / 100
            d = stats["D(%)"] / 100
            a = stats["A(%)"] / 100

            total = h + d + a
            h = round(h / total, 3)
            d = round(d / total, 3)
            a = round(a / total, 3)

            adjustment = 1 - (h + d + a)

            values = {"H": h, "D": d, "A": a}
            if adjustment != 0:
                key_to_adjust = max(values, key=lambda k: values[k] % 0.001)
                values[key_to_adjust] = round(values[key_to_adjust] + adjustment, 3)

            normalized_data[league][match] = values

            count += 1
            if count == 10:
                break
            
    return normalized_data

data_simulation = normalize_probabilities(data_prob)
save_json_file(output_path=f'{ parent_path }/database/json/simulation_probabilities.json', json_data=data_simulation)
#print(json.dumps(data_simulation, indent=4))