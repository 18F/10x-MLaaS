import pandas as pd
import os
import sys

class UpdateDb():

    def __init__(self):
        id_path = 'pastResponseId.txt'
        try:
            with open(id_path,'r') as f:
                    lines = f.read().splitlines()
                    try:
                        self.penultimateResponseId = lines[1]
                    except IndexError:
                        #if only 1 id, then there's no penultimate
                        self.penultimateResponseId = None
        except FileNotFoundError:
            self.penultimateResponseId = None


    def update_db(self):
        db_path = os.path.join('db','db.csv')
        db_df = pd.read_csv(db_path,encoding='latin1')
        if self.penultimateResponseId:
            idx = db_df.index[db['ResponseID'] == self.penultimateResponseId].tolist()[0]
            db_df = db_df.iloc[idx+1:]
        db_id_pred_map = dict(zip(db_df['ResponseID'], db_df['SPAM']))
        results_path = os.path.join('model','results','ClassificationResults.xlsx')
        results_df = pd.read_excel(results_path)
        validated_id_pred_map = dict(zip(results_df['ResponseID'],
                                         results_df['SPAM']))
        # For dictionaries db_id_pred_map and validated_id_pred_map, merged_id_pred_map
        # will be a merged dictionary with values from validated_id_pred_map replacing
        # those from db_id_pred_map
        merged_id_pred_map = {**db_id_pred_map, **validated_id_pred_map}
        num_merged_ids = len(merged_id_pred_map.keys())
        num_db_ids = len(db_id_pred_map.keys())

        if num_merged_ids > num_db_ids:
            for k in merged_id_pred_map:
                if k not in db_id_pred_map:
                    print(f"This ResponseID isn't in the db:  {f}.")
                    print("You shold probably look into this.")
            db_df['SPAM'] = db_df['ResponseID'].map(merged_id_pred_map)
        elif num_merged_ids < num_db_ids:
            print("The number of merged ids is less than the number of ids in the db.")
            print(f"This will fill {num_db_ids-num_merged_ids} ids with nan!")
            print("Exiting script")
            sys.exit(0)
        else:
            db_df['SPAM'] = db_df['ResponseID'].map(merged_id_pred_map)

        db_df.to_csv(db_path,index=False,encoding='latin1')
