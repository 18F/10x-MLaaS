import os
import pandas as pd
import dill as pickle
import sys
from utils.config import question_ids, survey_id
from datetime import datetime
from model.train import TrainClassifer



class MakePredictions():

    def __init__(self, df, survey_type = 'sw'):
        self.df = df
        if survey_type == 'sw':
            model_path = os.path.join(os.getcwd(), 'HSM', 'model','best_estimators','model_sw.pkl')
            self.model = model_path
        else:
            model_path = os.path.join(os.getcwd(), 'HSM', 'model','best_estimators','model_sw.pkl')
            self.model = model_path
 
    
    def prepare_data(self):
        df = self.df
        comments_concatenated = ''
        comments_original = ''
        for q in question_ids:
            comments_concatenated = comments_concatenated + " " + df[q].astype(str)
            comments_original = comments_original + "\n{}: ".format(q) + df[q].astype(str)

        df['Comments_Concatenated'] = comments_concatenated.apply(lambda x: x.strip())
        df['Normalized Comments'] = df['Comments_Concatenated'].apply(TrainClassifer().get_lemmas)
        X = df['Normalized Comments']
        response_ids = df['ResponseID']
        dates = df['EndDate']

        return X, response_ids, dates, comments_original


    def predict(self):
        outfile = 'ClassificationResults_{}_{}.xlsx'.format(survey_id, datetime.utcnow().strftime('%Y%m%d'))
        with open(self.model, 'rb') as f:
            pickled_model = pickle.load(f)
        X, response_ids, dates, comments_original = self.prepare_data()
        preds = pickled_model.predict(X)
        dec_func = pickled_model.decision_function(X)
        labeled_data_df = pd.DataFrame(X)
        labeled_data_df.columns = ['Comments Concatenated']
        labeled_data_df['SPAM'] = preds
        labeled_data_df['Decision Boundary Distance'] = abs(dec_func)
        labeled_data_df['ResponseID'] = response_ids
        labeled_data_df['Date'] = dates
        labeled_data_df['Original Survey Responses'] = comments_original
        results_dir = os.path.join(os.getcwd(),'model','results')
        if not os.path.exists(results_dir):
            os.makedirs(os.path.join(results_dir))
        results_path = os.path.join(results_dir, outfile)
        writer = pd.ExcelWriter(results_path)
        labeled_data_df.to_excel(writer, 'Classification Results', index=False)
        writer.save()
        id_pred_map = dict(zip(labeled_data_df['ResponseID'],
                               labeled_data_df['SPAM']))
        df = self.df.drop(labels=['Normalized Comments'], axis = 1)

        return results_path, df, id_pred_map, outfile
