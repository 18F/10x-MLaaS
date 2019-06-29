import pandas as pd


class Validate():

    def __init__(self, results_path):
        self.results_path = results_path

    def get_validations(self):
        '''
        Returns a mapping of responseIds to user-validated spam/ham codes
        '''
        validated_df = pd.read_excel(self.results_path)
        validated_id_pred_map = dict(zip(validated_df['ResponseID'],
                                         validated_df['SPAM']))
        return validated_id_pred_map
