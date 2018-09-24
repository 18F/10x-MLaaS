import pandas as pd
import os
import sys

class UpdateDb():

    def __init__(self):
        id_path = 'pastResponseId.txt'
        try:
            with open(id_path,'r') as f:
                lines = f.read().splitlines()
                if len(lines) == 1:
                    #if only 1 id, then None
                    self.lastResponseId = None
                else:
                    self.lastResponseId = lines[0]
        except FileNotFoundError:
            self.lastResponseId = None


    def update_db(self):
        db_path = os.path.join('db','db.csv')
        db_df = pd.read_csv(db_path,encoding='latin1')
        if self.lastResponseId:
            lastResponseId = self.lastResponseId
            idx = db_df.index[db_df['ResponseID'] == lastResponseId].tolist()[0]
            new_db_df = db_df.iloc[idx+1:]
            db_id_pred_map = dict(zip(new_db_df['ResponseID'], db_df['SPAM']))
        else:
            new_db_df = db_df
            db_id_pred_map = dict(zip(new_db_df['ResponseID'], db_df['SPAM']))

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
            new_db_df['SPAM'] = new_db_df['ResponseID'].map(merged_id_pred_map)
        elif num_merged_ids < num_db_ids:
            print("The number of merged ids is less than the number of ids in the db.")
            print(f"This will fill {num_db_ids-num_merged_ids} ids with nan!")
            print("Exiting script")
            sys.exit(0)
        else:
            new_db_df['SPAM'] = new_db_df['ResponseID'].map(merged_id_pred_map)

        if self.lastResponseId:
            concat_db = pd.concat([new_db_df,db_df])
            concat_db.to_csv(db_path,index=False,encoding='latin1')
        else:
            new_db_df.to_csv(db_path,index=False,encoding='latin1')
