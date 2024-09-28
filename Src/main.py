import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from tools import plot_and_table, normalize_data, load_json_file, save_json_file
from match import MatchRating
from combined_matchs import OneModel
from optimizer import OptimizerAdam

def main(league_name, match_rating_path):
    # Load existing results
    all_results = load_json_file(match_rating_path)
    
    # Train
    file_train = fr'D:\LUCAS\Match Rating\Database\{ league_name }\train'
    datas_train = os.listdir(file_train)

    matchs_rating = {
        'Gols':{},
        'Target Shoots': {}
    }
    for data in datas_train:
        df = pd.read_csv(os.path.join(file_train, data))
        for stats in ['Gols','Target Shoots']:
            test = MatchRating(matchs_rating=matchs_rating, estatistic=stats)
            test.get_columns()
            test.get_match_rating(data=df)
            
    results_gols = plot_and_table(data_dict=dict(sorted(matchs_rating['Gols'].items())), range=(-28,28), show_grap=False)
    results_ts = plot_and_table(data_dict=dict(sorted(matchs_rating['Target Shoots'].items())), range=(-45,61), show_grap=False)
    
    # Initialize league results if not present
    if league_name not in all_results:
        all_results[league_name] = {}
        
    # Update league results
    all_results[league_name].update({
        'w1': [],
        'Gols': results_gols,
        'Target Shoots': results_ts
    })
    
    # Test
    file_test = fr'D:\LUCAS\Match Rating\Database\{ league_name }\test'
    datas_test = os.listdir(file_test)
    
    optimizer = OptimizerAdam(learning_rate=0.001)
    
    w1 = 0.55 # Chute inicial
    for data in datas_test:
        df = pd.read_csv(os.path.join(file_test, data))
        
        onemodel = OneModel(data=df, models_ratings=all_results[league_name])
        w1 = onemodel.get_match_rating(w1=w1, optimizer=optimizer)
    
    all_results[league_name]['w1'] = w1
    
    # Save updated results to JSON file
    save_json_file(match_rating_path, all_results)
    
if __name__ == '__main__':
    for league in ['Premier League', 'La Liga']:
        main(league_name=league, match_rating_path=r'D:\LUCAS\Football_2.0\Database\Static\matchs_ratings.json')