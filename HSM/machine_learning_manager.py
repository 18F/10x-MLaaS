import os
import pandas as pd
import dill as pickle
from datetime import datetime
from model.train import TrainClassifier
from utils.config import FIELDS, survey_id
import storage


class MachineLearningManager():
    def __init__(self, dataframe, path, space):
        self.data = dataframe
        self.path = path
        self.model = None
        self.space = space

    def prepare(self):
        pass

    def get_model(self):
        # local
        if self.model is None:
            if (self.space == 'local'):
                with open(self.path, 'rb') as f:
                    self.model = pickle.load(f)
            elif (self.space == 's3'):
                with storage.S3File(storage.S3Storage(), self.path) as f:
                    self.model = pickle.load(f)

        return self.model

    def save_model(self):
        pass

    def _prepare_data(self, data):
        # This is very specific to the survey and will need to be abstract away
        df = data
        # Because these types are python objects, they need to be converted.
        other_purpose = df['Q5'].astype(str)
        # print("1")
        unable_complete = df['Q7'].astype(str)
        # print("2")
        value = df['Q6'].astype(str)
        # print("3")
        purpose = df['Q3'].astype(str)
        # print("4")
        comments_concatenated = other_purpose+" "+unable_complete+" "+value+" "+purpose
        # print("5")
        comments_original = "Q3: " + purpose + "\n Q5: " + other_purpose + "\n Q6: " + value + "\n Q7: " \
            + unable_complete
        # print("6")
        df['Comments_Concatenated'] = comments_concatenated.apply(lambda x: x.strip())
        # print("7")
        df['Normalized Comments'] = df['Comments_Concatenated'].apply(TrainClassifier().get_lemmas)
        # print("8")
        X = df['Normalized Comments']
        # print("9")
        response_ids = df['ResponseID']
        # print("10")
        dates = df['EndDate']

        # print("11")
        print("done preparing")
        return X, response_ids, dates, comments_original

    def predict(self, data):
        self.data = data
        pickled_model = self.get_model()
        print("got the model")
        print(pickled_model)
        X, response_ids, dates, comments_original = self._prepare_data(data)
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
        print(list(self.data))
        print("Here's the list of available items to choose from the processed prediction data:")
        print(list(labeled_data_df))
        # Using Outer Join to get all the data even if there's missing info on one side
        joined_df = pd.merge(data, labeled_data_df, on='ResponseID', how='outer')
        joined_df = pd.merge(self.data, labeled_data_df, on='ResponseID', how='outer')

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

        # Only pick out what the user wants to output + the Decision Boundary Distance
        joined_df = joined_df[valid_fields + ["Decision Boundary Distance"]]
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
        df = self.data.drop(labels=['Normalized Comments'], axis=1)

        return results_path, df, id_pred_map, outfile
