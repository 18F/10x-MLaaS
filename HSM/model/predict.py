import os
import pandas as pd
import dill as pickle
from datetime import datetime
from model.train import TrainClassifier
from utils.config import FIELDS, survey_id


class MakePredictions():

    def __init__(self, df, survey_type='sw'):
        self.df = df
        if survey_type == 'sw':
            model_path = os.path.join(os.getcwd(), 'HSM', 'model', 'best_estimators', 'model_sw.pkl')
            self.model = model_path
        # TODO: This two paths look the same, what was the intention here?
        else:
            model_path = os.path.join(os.getcwd(), 'HSM', 'model', 'best_estimators', 'model_sw.pkl')
            self.model = model_path

    def prepare_data(self):
        df = self.df
        # Because these types are python objects, they need to be converted.
        other_purpose = df['Q5'].astype(str)
        unable_complete = df['Q7'].astype(str)
        value = df['Q6'].astype(str)
        purpose = df['Q3'].astype(str)
        comments_concatenated = other_purpose+" "+unable_complete+" "+value+" "+purpose
        comments_original = "Q3: " + purpose + "\n Q5: " + other_purpose + "\n Q6: " + value + "\n Q7: " \
            + unable_complete
        df['Comments_Concatenated'] = comments_concatenated.apply(lambda x: x.strip())
        df['Normalized Comments'] = df['Comments_Concatenated'].apply(TrainClassifier().get_lemmas)
        X = df['Normalized Comments']
        response_ids = df['ResponseID']
        dates = df['EndDate']

        return X, response_ids, dates, comments_original

    def predict(self):
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

        print("Combining all specified columns and prediction...")
        print("Here's the list of available items to choose from the raw data:")
        print(list(self.df))
        print("Here's the list of available items to choose from the processed prediction data:")
        print(list(labeled_data_df))
        # Using Outer Join to get all the data even if there's missing info on one side
        joined_df = pd.merge(self.df, labeled_data_df, on='ResponseID', how='outer')

        # There are two SPAM columns, SPAM_x and SPAM_y, SPAM_x should be removed becauses it is an empty column
        # Need to rename SPAM_y to SPAM
        joined_df = joined_df.drop(columns='SPAM_x')
        joined_df = joined_df.rename(columns={'SPAM_y': 'SPAM'})

        # Try to figure out what is valid columns
        valid_fields = [field for field in FIELDS if field in list(joined_df)]

        if valid_fields != FIELDS:
            invalid_fields = [field for field in FIELDS if field not in list(joined_df)]
            print("Here is a list of fields that cannot be found in the data:")
            print(invalid_fields)
            print("Skipping them in the output...")

        # Only pick out what the user wants to output
        joined_df = joined_df[valid_fields]
        print("Here is the final list of the valid user-choosen fields in the config.py file we are using.")
        print(list(valid_fields))

        results_dir = os.path.join(os.getcwd(), 'model', 'results')
        if not os.path.exists(results_dir):
            os.makedirs(os.path.join(results_dir))
        outfile = 'ClassificationResults_{}_{}.xlsx'.format(survey_id, datetime.now().strftime('%Y%m%d-%H%M%S'))
        results_path = os.path.join(results_dir, outfile)
        writer = pd.ExcelWriter(results_path)
        # labeled_data_df.to_excel(writer, 'Classification Results', index=False)
        joined_df.to_excel(writer, 'Classification Results', index=False)
        writer.save()
        id_pred_map = dict(zip(labeled_data_df['ResponseID'],
                               labeled_data_df['SPAM']))
        df = self.df.drop(labels=['Normalized Comments'], axis=1)

        return results_path, df, id_pred_map, outfile
