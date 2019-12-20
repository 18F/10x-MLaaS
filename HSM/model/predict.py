import os
import pandas as pd
import dill as pickle
from datetime import datetime
from model.train import TrainClassifier
from utils.config import (
    ENTRY_ID,
    FIELDS,
    FIELDS_TO_INCLUDED_FOR_PROCESSED_DATA_MAPPING,
    FILTER_FEATURE,
    FILTER_FEATURE_FIELDS,
    NORMALIZED_FILTER_FEATURE,
    PREDICTION_FIELD_NAME,
    survey_id,
)


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
        comments_concatenated = ""
        comments_original = ""

        for field in FILTER_FEATURE_FIELDS:
            # Because these types are python objects, they need to be converted.
            value = df[field].astype(str)
            comments_concatenated += value
            comments_original += f'{field}: {value}\n'
        df['Comments_Concatenated'] = comments_concatenated.apply(lambda x: x.strip())
        df[NORMALIZED_FILTER_FEATURE] = df['Comments_Concatenated'].apply(TrainClassifier().get_lemmas)
        X = df[NORMALIZED_FILTER_FEATURE]

        result_series = {s: df[FIELDS_TO_INCLUDED_FOR_PROCESSED_DATA_MAPPING[s]]
                         for s in FIELDS_TO_INCLUDED_FOR_PROCESSED_DATA_MAPPING}

        result_series['Original Survey Responses'] = comments_original

        return X, result_series

    def predict(self):
        with open(self.model, 'rb') as f:
            pickled_model = pickle.load(f)
        X, data_columns = self.prepare_data()
        preds = pickled_model.predict(X)
        dec_func = pickled_model.decision_function(X)
        labeled_data_df = pd.DataFrame(X)
        labeled_data_df.columns = [FILTER_FEATURE]
        labeled_data_df[PREDICTION_FIELD_NAME] = preds
        labeled_data_df['Decision Boundary Distance'] = abs(dec_func)
        for col in data_columns:
            labeled_data_df[col] = data_columns[col]

        print("Combining all specified columns and prediction...")
        print("Here's the list of available items to choose from the raw data:")
        print(list(self.df))
        print("Here's the list of available items to choose from the processed prediction data:")
        print(list(labeled_data_df))
        # Using Outer Join to get all the data even if there's missing info on one side
        joined_df = pd.merge(self.df, labeled_data_df, on=ENTRY_ID, how='outer')

        if PREDICTION_FIELD_NAME + '_x' in joined_df.columns:  # This means there are two SPAM columns
            # There can be two columns for the prediction fields, one in df (which will have a suffix of _x),
            # one in labeled_data_df (which will have a suffix of _y),
            # and we will keep the one from labeled_data_df because it holds the actual prediction
            # but this field needs to rename to with _y.
            # i.e. SPAM is the field name and it appears in df and labeled_data_df, then we will have SPAM_x
            # and SPAM_y column when joined in joined_df, right now assuming we are using SPAM_y because it
            # holds the actual prediction, and SPAM_x should be removed because it came from the raw data.
            # We also need to rename SPAM_y to SPAM
            joined_df = joined_df.drop(columns=PREDICTION_FIELD_NAME + '_x')
            joined_df = joined_df.rename(columns={PREDICTION_FIELD_NAME + '_y': PREDICTION_FIELD_NAME})

        # Try to figure out what is valid columns
        valid_fields = [field for field in FIELDS if field in list(joined_df)]

        if valid_fields != FIELDS:
            invalid_fields = [field for field in FIELDS if field not in list(joined_df)]
            print("Here is a list of fields that cannot be found in the data:")
            print(invalid_fields)
            print("Skipping them in the output...")

        # Only pick out what the user wants to output + the filter feature and normalized filter feature,
        # and the Decision Boundary Distance
        joined_df = joined_df[valid_fields + [FILTER_FEATURE, NORMALIZED_FILTER_FEATURE, "Decision Boundary Distance"]]
        print("Here is the final list of the valid user-choosen fields in the config.py file we are using.")
        print(list(valid_fields))

        results_dir = os.path.join(os.getcwd(), 'HSM', 'model', 'results')
        if not os.path.exists(results_dir):
            os.makedirs(os.path.join(results_dir))
        outfile = 'ClassificationResults_{}_{}.xlsx'.format(survey_id, datetime.now().strftime('%Y%m%d-%H%M%S'))
        results_path = os.path.join(results_dir, outfile)
        writer = pd.ExcelWriter(results_path)
        joined_df.to_excel(writer, 'Classification Results', engine='xlsxwriter', index=False)
        writer.save()
        id_pred_map = dict(zip(labeled_data_df[ENTRY_ID],
                               labeled_data_df[PREDICTION_FIELD_NAME]))
        df = self.df.drop(labels=[NORMALIZED_FILTER_FEATURE], axis=1)

        return results_path, df, id_pred_map, outfile
