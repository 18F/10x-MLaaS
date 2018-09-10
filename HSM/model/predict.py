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
from model.text_utils import NormalizeText

warnings.filterwarnings('ignore')


class ClassifyNewData():

    def __init__(self, db_path, model):
        self.db_path = db_path
        if model == 'Value':
            model_name = model+'test_roc_auc_best_estimator.pkl'
            model_path = os.path.join("/Users",'scottmcallister','Desktop','GitHub','10x-qualitative-data','HSM','model','best_estimators',model_name)
            self.model = model_path
        else:
            self.model = None
        id_path = os.path.join("/Users",'scottmcallister','Desktop','GitHub','10x-qualitative-data','HSM','pastResponseId.txt')
        try:
            with open(id_path,'r') as f:
                    lines = f.read().splitlines()
                    try:
                        self.penultimateResponseId = lines[1]
                    except IndexError:
                        #if only 1 id, then there's no penultimate
                        self.penultimateResponseId = None
        except FileNotFoundError:
            print("Can't predict without data. Run qualtrics.py first.")
            sys.exit(0)

    def get_new_data(self):
        db = pd.read_csv(db_path)
        if self.penultimateResponseId:
            idx = db.index[db['ResponseID'] == self.penultimateResponseId].tolist()[0]
            db = db.iloc[idx+1:]
        if db.size == 0:
            print("No new data to predict on!")
            sys.exit(0)
        nt = NormalizeText()
        normalized_text = nt.transform(db['Value'])
        db['Normalized Value'] = normalized_text
        db = db.dropna(subset=['Value'])
        ordinal_questions = ["Experience Rating",
                             "Likely to Return",
                             "Likely to Recommend",
                             "Able to Accomplish"
                             ]
        for col in ordinal_questions:
            if col=='Likely to Return' or col=="Likely to Recommend":
                db[col]= db[col].map({1:'Very unlikely',
                                      2:'Unlikely',
                                      3:'Neither likely nor unlikely',
                                      4:'Likely',
                                      5:'Very likely'})
            elif col=="Able to Accomplish":
                db[col] = db[col].map({1:'No',
                                       2:'Not yet, but still trying',
                                       3:'Just browsing / not trying to accomplish anything specific',
                                       4:'Yes, partly',
                                       5:'Yes, fully',
                                       6:'Yes, fully'})
            else:
                db[col] = db[col].map({1:'Very poor',
                                       2:'Poor',
                                       3:'Fair',
                                       4:'Good',
                                       5:'Very good'})
        return db

    def predict(self):
        with open(self.model, 'rb') as f:
            pickled_model = pickle.load(f)
        new_data = self.get_new_data()
        preds = pickled_model.predict(new_data)
        try:
            pred_probs = pickled_model.predict_proba(new_data)
            new_data['Value Spam'] = preds
            new_data['Ham Prediction Probability'] = 0
            new_data['Spam Prediction Probability'] = 0
            new_data[['Ham Prediction Probability',
                      'Spam Prediction Probability']] = pd.DataFrame(pred_probs).values
            pred_prob_deltas = abs(new_data['Ham Prediction Probability'] - new_data['Spam Prediction Probability'])
            new_data['Value Prediction Probabilities Delta'] = pred_prob_deltas
            new_data = new_data.drop(labels=['Ham Prediction Probability',
                                             'Spam Prediction Probability'],
                                     axis=1)
        except AttributeError:
            #catch models that do not have the predict_proba() method
            dec_func = pickled_model.decision_function(new_data)
            new_data['Value Spam'] = preds
            new_data['Value Decision Boundary Distance'] = abs(dec_func)
        results_dir = os.path.join(os.getcwd(),'results')
        if not os.path.exists(results_dir):
            os.makedirs(os.path.join(results_dir))
        results_path = os.path.join(results_dir,'ClassificationResults.xlsx')
        writer = pd.ExcelWriter(results_path)
        new_data.to_excel(writer,'Classification Results',index=False)
        writer.save()


db_path = os.path.join("/Users",'scottmcallister','Desktop','GitHub','10x-qualitative-data','HSM','db','db.csv')
nd = ClassifyNewData(db_path,'Value')
nd.get_new_data()
nd.predict()
print("Done making predictions. You can find the results in ClassificationResults.xlsx")
