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

warnings.filterwarnings('ignore')

# These might need to stay global until I can figure out how to include in the sklearn Pipeline.
nlp = spacy.load('en', parse = False, tag=False, entity=False)
tokenizer = ToktokTokenizer()
stopword_list = stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def replace_accented_chars(text):
    #The normal form KD (NFKD) will apply the compatibility decomposition,
    #i.e. replace all compatibility characters with their equivalents (from python.org).
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text, contraction_mapping=contractions.contractions_dict):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        if contraction_mapping.get(match):
            expanded_contraction = contraction_mapping.get(match)
        else:
            expanded_contraction = contraction_mapping.get(match.lower())
        if expanded_contraction:
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction
        else:
            pass

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_corpus(doc, html_stripping=True, contraction_expansion=True, text_lemmatization=True,
                     stopword_removal=True):

    def get_profanity():
        file_path = os.path.join(os.getcwd(),"corpora","profanity.csv")
        profanity = set(pd.read_csv(file_path).values.ravel().tolist())
        return profanity

    #url regex
    url_re = re.compile(r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")
    #email address regex
    email_re = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')
    #phone number regex
    phone_re = re.compile(r'(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?')
    #ssn regex
    ssn_re = re.compile(r'^(?!219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}[-]?(?!00)\d{2}[-]?(?!0{4})\d{4}$')
    #profanity regex
    profanity_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, get_profanity())))

    if doc is np.nan:
        return ""
    else:
        doc = doc.lower()

        doc = profanity_regex.sub("criticaster", doc)
        doc = email_re.sub('blatherskite',doc)
        doc = phone_re.sub('blatherskite',doc)
        doc = ssn_re.sub('blatherskite',doc)
        doc = url_re.sub('blatherskite',doc)

        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)
        # at least three characters long, cannot contain a number, and no more than 17 chars long
        doc = re.findall(r'\b[a-z][a-z][a-z]+\b',doc)
        doc = ' '.join(w for w in doc if w != 'nan' and len(w) <= 17)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=True)
        if len(doc) == 0:
            doc = "spam"
        return doc


class ClassifyNewData():

    def __init__(self, db_path, normalize_corpus, model):
        self.db_path = db_path
        self.normalize_corpus = normalize_corpus
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
        db['Normalized Value'] = db['Value'].apply(self.normalize_corpus)
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
nd = ClassifyNewData(db_path, normalize_corpus,'Value')
nd.get_new_data()
nd.predict()
print("Done making predictions. You can find the results in ClassificationResults.xlsx")
