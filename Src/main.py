import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from tools import plot_and_table, normalize_data, save_json_file
from match import MatchRating
from combined_matchs import OneModel
from optimizer import OptimizerAdam

def main():
     # Train
    file_train = r'D:\LUCAS\Match Rating\Database\Premier League\train'
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
    
    all_results = {
        'Gols': results_gols,
        'Target Shoots': results_ts
    }
    
    save_json_file(
        output_path=r'D:\LUCAS\Football_2.0\Database\Static\matchs_ratings.json',
        json_data=all_results
    )
    
    # # Test
    # file_test = r'D:\LUCAS\Match Rating\Database\Premier League\test'
    # datas_test = os.listdir(file_test)
    
    # optimizer = OptimizerAdam(learning_rate=0.005)
    
    # w1 = 0.6 # Chute inicial
    # for data in datas_test:
    #     df = pd.read_csv(os.path.join(file_test, data))
        
    #     onemodel = OneModel(data=df, models_ratings=all_results)
    #     w1 = onemodel.get_match_rating(w1=w1, optimizer=optimizer)
    
    
if __name__ == '__main__':
    main()