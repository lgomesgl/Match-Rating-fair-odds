import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from tools import normalize_data, load_json_file, save_json_file
from match import MatchRating
from regression_polynomial import RegressionPolynomial
from combined_matchs import OneModel
from optimizer import OptimizerAdam

def main(league_name, match_rating_path):
    logging.info(f'Start into { league_name }')
    
    # Load existing results
    all_results = load_json_file(match_rating_path)
    
    logging.info(f'Start into train data')
    # Train
    file_train = fr'D:\LUCAS\Match Rating\Database\{ league_name }\train'
    datas_train = os.listdir(file_train)

    matchs_rating = {
        'Gols':{},
        'Shoots':{},
        'Target Shoots':{}
    }
    for data in datas_train:
        df = pd.read_csv(os.path.join(file_train, data))
        for stats in ['Gols','Shoots','Target Shoots']:
            match_rat = MatchRating(matchs_rating=matchs_rating, estatistic=stats, gols=1.5)
            match_rat.get_columns()
            match_rat.get_match_rating(data=df) 
    
    rp_gols = RegressionPolynomial(league_name=league_name,
                                        stats='Gols',
                                        match_rating=dict(sorted(matchs_rating['Gols'].items())),
                                        range=(-29,28))
    results_gols = rp_gols.fit(show_graphs=False) 
    
    rp_shoots = RegressionPolynomial(league_name=league_name,
                                        stats='Shoots',
                                        match_rating=dict(sorted(matchs_rating['Shoots'].items())),
                                        range=(-50,50))
    results_shoots = rp_shoots.fit(show_graphs=False)   
     
    rp_ts = RegressionPolynomial(league_name=league_name,
                                        stats='Target Shoots',
                                        match_rating=dict(sorted(matchs_rating['Target Shoots'].items())),
                                        range=(-57,61))
    results_ts = rp_ts.fit(show_graphs=False)                              
    
    # Initialize league results if not present
    if league_name not in all_results:
        all_results[league_name] = {}
        
    # Update league results
    all_results[league_name].update({
        'w1': [],
        'Gols': results_gols,
        'Shoots': results_shoots,
        'Target Shoots': results_ts
    })
    
    # Test
    logging.info(f'Start into test data')
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
    for league in ['Premier League', 'La Liga', 'Bundesliga', 'SerieA', 'Ligue1']:
        main(league_name=league, match_rating_path=r'D:\LUCAS\Football_2.0\Database\Static\matchs_ratings.json')
         