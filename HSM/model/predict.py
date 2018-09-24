import os
import pandas as pd
import dill as pickle
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import contractions
from bs4 import BeautifulSoup
import unicodedata
import spacy
import re
import sys
import numpy as np
from math import sin, cos, pi
import warnings
from model.train import TrainClassifer

warnings.filterwarnings('ignore')


class ClassifyNewData():

    def __init__(self, db_path, survey_type = 'sw'):
        self.db_path = db_path
        if survey_type == 'sw':
            model_name = 'model_sw.pkl'
            model_path = os.path.join(os.getcwd(),'model','best_estimators',model_name)
            self.model = model_path
        else:
            model_name = 'model_pl.pkl'
            model_path = os.path.join(os.getcwd(),'model','best_estimators',model_name)
            self.model = model_path
        id_path = os.path.join(os.getcwd(),'pastResponseId.txt')
        try:
            with open(id_path,'r') as f:
                    lines = f.read().splitlines()
                    try:
                        self.penultimateResponseId = lines[1]
                        print("here!")
                    except IndexError:
                        #if only 1 id, then None
                        self.penultimateResponseId = None
        except FileNotFoundError:
            print("Can't predict without data. Run qualtrics.py first.")
            sys.exit(0)


    def get_new_data(self):
        db = pd.read_csv(self.db_path,encoding='latin1')
        if self.penultimateResponseId:
            # get new responses using penultimateResponseId
            idx = db.index[db['ResponseID'] == self.penultimateResponseId].tolist()[0]
            new_data = db.iloc[idx+1:]
        else:
            new_data = db
        if new_data.size == 0:
            print("No new data to predict on! Exiting the script. Try again later.")
            sys.exit(0)
        other_purpose = new_data['Other Purpose of Visit'].astype(str)
        unable_complete = new_data['Unable to Complete Purpose Reason'].astype(str)
        value = new_data['Value'].astype(str)
        purpose = new_data['Purpose'].astype(str)
        new_data['Comments Concatenated'] = other_purpose+" "+unable_complete+" "+value+" "+purpose
        new_data['Normalized Comments'] = new_data['Comments Concatenated'].apply(TrainClassifer().get_lemmas)
        X = new_data['Normalized Comments']
        response_ids = new_data['ResponseID']
        return X, response_ids


    def predict(self):
        with open(self.model, 'rb') as f:
            pickled_model = pickle.load(f)
        new_data, response_ids = self.get_new_data()
        preds = pickled_model.predict(new_data)
        dec_func = pickled_model.decision_function(new_data)
        labeled_data_df = pd.DataFrame(new_data)
        labeled_data_df.columns = ['Comments Concatenated']
        labeled_data_df['SPAM'] = preds
        labeled_data_df['Decision Boundary Distance'] = abs(dec_func)
        labeled_data_df['ResponseID'] = response_ids
        results_dir = os.path.join(os.getcwd(),'model','results')
        if not os.path.exists(results_dir):
            os.makedirs(os.path.join(results_dir))
        results_path = os.path.join(results_dir,'ClassificationResults.xlsx')
        writer = pd.ExcelWriter(results_path)
        labeled_data_df.to_excel(writer,'Classification Results',index=False)
        writer.save()
