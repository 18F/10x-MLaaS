import contractions
import numpy as np
import re
import os
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from scipy import stats
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# in order to use SMOTE, you've got to import Pipeline from imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import dill as pickle
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


class log_uniform():
    """
    Provides an instance of the log-uniform distribution with an .rvs() method.
    Meant to be used with RandomizedSearchCV, particularly for hyperparams like
    alpha, C, gamma, etc.

    Attributes:
        a (int or float): the exponent of the beginning of the range and
        b (int or float): the exponent of the end of range.
        base (int or float): the base of the logarithm. 10 by default.
    """

    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=1, random_state=None):
        uniform = stats.uniform(loc=self.loc, scale=self.scale)
        return np.power(self.base,
                        uniform.rvs(size=size,
                                    random_state=random_state))


class TrainClassifier():
    """
    Description:
        This class will train a model depending for the site-wide survey.
    Attributes:
        metric (str): the classifier scoring metric to use. Choose from:
        accuracy, roc_auc, avg_precision, fbeta, or recall. Note that for fbeta,
        beta = 2.
    """

    def __init__(self, metric='avg_precision'):
        self.metric = metric

    @staticmethod
    def clean(doc):
        """
        Prepares text for NLP by stripping html tags; replacing urls with 'url';
        and replacing email addresses with 'email'. It also expands contractions
        and lowercases everything. Finally, it only keeps words that are at least
        three characters long, do not contain a number, and are no more than
        17 chars long.

        Arguments:
            doc (str): A single document within the corpus.

        Returns:
            normalized (str): The normalized string.
        """

        def strip_html_tags(text):
            """
            Strips html tags from a string.
            """

            soup = BeautifulSoup(text, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        def strip_urls(text):
            """
            Replaces urls in a string with 'url'.
            """
            # @TODO: This needs more comments and explanations
            pattern = r"""
                (?i)\b    # There is 0 boundary
                ((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/) # This is the base address
                (?:[^\s()<>]+|
                \(([^\s()<>]+|
                (\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|
                (\([^\s()<>]+\)))*\)|
                [^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""

            url_re = re.compile(pattern, re.VERBOSE)
            text = url_re.sub('url', text)
            return text

        def strip_emails(text):
            """
            Replaces email addresses in a string with 'email'.
            """

            email_re = re.compile(r'\S+@\S+')
            text = email_re.sub('email', text)
            return text

        def strip_nonsense(text):
            """
            Returns words from a string that are at least 3 characters long, do not contain a number, and
            are no more than 17 chars long.
            """

            no_nonsense = re.findall(r'\b[a-z][a-z][a-z]+\b', text)
            text = ' '.join(w for w in no_nonsense if w != 'nan' and len(w) <= 17)
            return text

        def expand_contractions(text, contraction_mapping=contractions.contractions_dict):
            """
            Expands contractions within a string. For example, can't becomes cannot.
            """

            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                              flags=re.IGNORECASE | re.DOTALL)

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

        doc = doc.lower()
        contraction_free = expand_contractions(doc)
        tag_free = strip_html_tags(contraction_free)
        url_free = strip_urls(tag_free)
        email_free = strip_emails(url_free)
        normalized = strip_nonsense(email_free)
        return normalized

    @staticmethod
    def get_lemmas(document):
        """
        Lemmatizes the string of a single document after normalizing it with the
        clean function.

        Arguments:
            document (str): A single document within the corpus.

        Returns:
            lemmas_str (str): A space-delimited string of lemmas. This can be
                              passed into a word vectorizer, such as tf-idf.
        """

        def get_wordnet_pos(treebank_tag):
            """
            Converts the part of speech tag returned by nltk.pos_tag() to a value
            that can be passed to the `pos` kwarg of wordnet_lemmatizer.lemmatize()
            """

            if treebank_tag.startswith('J'):
                return wn.ADJ
            elif treebank_tag.startswith('V'):
                return wn.VERB
            elif treebank_tag.startswith('N'):
                return wn.NOUN
            elif treebank_tag.startswith('R'):
                return wn.ADV
            else:
                return wn.NOUN

        # stopword_set = set(stopwords.words('english'))
        # using the clean function defined above here
        text = word_tokenize(TrainClassifier.clean(document))
        word_pos = nltk.pos_tag(text)
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word, pos in word_pos:
            pos = get_wordnet_pos(pos)
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            if 'research' in lemma:
                lemmas.append('research')
            elif 'dataset' in lemma:
                lemmas.append('dataset')
            else:
                lemmas.append(lemma)
        # lemmas_list = [lemma for lemma in lemmas if lemma not in stopword_set]
        lemmas_str = " ".join(lemma for lemma in lemmas)
        return lemmas_str

    def prepare_train(self):
        labeled_data_path = os.path.join('model',
                                         'training_data',
                                         'training-sw.xlsx')

        train_df = pd.read_excel(labeled_data_path)
        print("\tNormalizing the text...")
        # normalize the comments, preparing for tf-idf
        train_df['Normalized Comments'] = train_df['Comments Concatenated'].astype(str).apply(
            TrainClassifier.get_lemmas)
        print("\tDone normalizing the text.")
        print("_"*80)
        return train_df

    def randomized_grid_search(self,
                               train_df,
                               clf=SGDClassifier(),
                               n_iter_search=10,  # 10 for testing purposes
                               pickle_best=True):
        """
        Given labeled training data (`df`) for a binary classification task,
        performs a randomized grid search `n_iter_search` times using `clf` as the
        classifier and the `score` as a scoring metric.

        Attributes:
            df (pandas DataFrame):  The training data. Currently, you must specify
                                    within the function the label and feature column
                                    names.
            clf (instance of an sklearn classifier):  SGDClassifier() by default
            n_iter_search:  number of parameter settings that are sampled. Trades
                            off runtime vs quality of the solution.
            pickle_best (bool): whether or not to pickle the best estimator
                                returned by the grid search. Default is True
        """

        score = self.metric
        scoring = {'accuracy': metrics.make_scorer(metrics.accuracy_score),
                   'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
                   'avg_precision': metrics.make_scorer(metrics.average_precision_score),
                   'fbeta': metrics.make_scorer(metrics.fbeta_score, beta=1.5),
                   'recall': metrics.make_scorer(metrics.recall_score)}
        # clf_name = clf.__class__.__name__
        X = train_df['Normalized Comments']
        # y = train_df['Spam']
        y = train_df['SPAM']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=123)
        pipe = Pipeline([
                         ('vectorizer', TfidfVectorizer()),
                         ('upsample', SMOTE()),
                         ('select', SelectPercentile()),
                         ('clf', clf)])
        param_dist = {
                      "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
                      "vectorizer__min_df": stats.randint(1, 3),
                      "vectorizer__max_df": stats.uniform(.7, .3),
                      "vectorizer__sublinear_tf": [True, False],
                      "upsample": [None,
                                   SMOTE(ratio='minority', kind='svm'),
                                   SMOTE(ratio='minority', kind='regular'),
                                   SMOTE(ratio='minority', kind='borderline1'),
                                   SMOTE(ratio='minority', kind='borderline2')],
                      "select": [None,
                                 SelectPercentile(percentile=10),
                                 SelectPercentile(percentile=20),
                                 SelectPercentile(percentile=50),
                                 SelectPercentile(percentile=75)],
                      "clf__alpha": log_uniform(-5, 2),
                      "clf__penalty": ['l2', 'l1', 'elasticnet'],
                      "clf__loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                      }

        random_search = RandomizedSearchCV(pipe, param_distributions=param_dist,
                                           scoring=scoring, refit=score,
                                           n_iter=n_iter_search, cv=5, n_jobs=-1,
                                           verbose=1)
        random_search.fit(X_train, y_train)
        y_pred = random_search.predict(X_test)
        # get the col number of the positive class (i.e. spam)
        positive_class_col = list(random_search.classes_).index(1)
        try:
            y_score = random_search.predict_proba(X_test)[:, positive_class_col]
        except AttributeError:
            y_score = random_search.decision_function(X_test)
        average_precision = metrics.average_precision_score(y_test, y_score)
        acc = metrics.accuracy_score(y_test, y_pred)
        roc_auc = metrics.roc_auc_score(y_test, y_pred)
        precisions, recalls, _ = metrics.precision_recall_curve(y_test, y_score)
        auc = metrics.auc(recalls, precisions)
        fbeta = metrics.fbeta_score(y_test, y_pred, beta=1.5)
        recall = metrics.recall_score(y_test, y_pred)
        print("\tRecall on test data:  {0:.2f}".format(recall))
        print("\tAccuracy on test data:  {0:.2f}".format(acc))
        print("\tROC-AUC on test data:  {0:.2f}".format(roc_auc))
        print("\tFbeta on test data:  {0:.2f}".format(fbeta))
        print("\tAverage Precision on test data:  {0:.2f}".format(average_precision))
        print("\tPrecision-Recall AUC on test data:  {0:.2f}".format(auc))
        print("-"*80)
        print("Classification Report:")
        class_names = ['ham', 'spam']
        print(metrics.classification_report(y_test,
                                            y_pred,
                                            target_names=class_names))
        best_estimator = random_search.best_estimator_
        best_score = random_search.best_score_
        result_values = [y_pred, y_score, precisions, recall, average_precision,
                         acc, roc_auc, auc, fbeta, recalls, best_score, best_estimator, y_test]
        result_keys = ['y_pred', 'y_score', 'precisions', 'recall', 'average_precision', 'acc',
                       'roc_auc', 'auc', 'fbeta', 'recalls', 'best_score', 'best_estimator', 'y_test']
        results = {k: v for k, v in zip(result_keys, result_values)}
        if pickle_best:
            pickle_dir = os.path.join(os.getcwd(), 'model', 'best_estimators')
            if not os.path.exists(pickle_dir):
                os.makedirs(pickle_dir)
            pickle_path = os.path.join(pickle_dir, 'model_sw.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(random_search.best_estimator_, f)
        return results


if __name__ == '__main__':
    tc = TrainClassifier()
    train_df = tc.prepare_train()
    results = tc.randomized_grid_search(train_df)
