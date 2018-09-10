import numpy as np
import pandas as pd
import numpy as np
import gensim
from collections import defaultdict
from math import sin, cos, pi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.downloader as api
import os


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """
    Description:
        This class is designed for use as a transformer within an sklearn
        pipeline. The pipeline will call the fit and transform instance methods.

        The transform method will simply select the columns provided to __init__.
    """

    def __init__(self, cols, dtype = 'float'):
        self.dtype = dtype
        self.cols = cols


    def transform(self, X_train):
        col_list = []
        for c in self.cols:
            if self.dtype == 'float':
                col_list.append(X_train[c].values)
            elif self.dtype == 'str':
                col_list.append(X_train[c].astype(str))
            elif self.dtype == 'datetime':
                col_list.append(pd.to_datetime(X_train[c]))

        if len(col_list) == 1:
            return col_list[0]
        else:
            return pd.concat(col_list,axis=1)


    def fit(self, X_train, y=None):
        return self


class OrdinalTransformer(BaseEstimator, TransformerMixin):
    """
    Description:
        This class is designed for use as a transformer within an sklearn
        pipeline. The pipeline will call the fit and transform instance methods.

        The transform method maps the likert respones to numerics.
    """


    def __init__(self, cols):
        self.cols = cols


    def transform(self, X_train, y=None):
        if len(self.cols)>1:
            ordinal_data = []
            for col in self.cols:
                if col=='Likely to Return':
                    ret= X_train[col].map({'Very unlikely':1,
                                           'Unlikely':2,
                                           'Neither likely nor unlikely':3,
                                           'Likely':4,
                                           'Very likely':5})
                    ordinal_data.append(ret)
                elif col=="Likely to Recommend":
                    rec = X_train[col].map({'Very unlikely':1,
                                            'Unlikely':2,
                                            'Neither likely nor unlikely':3,
                                            'Likely':4,
                                            'Very likely':5})
                    ordinal_data.append(rec)
                elif col=="Able to Accomplish":
                    acc = X_train[col].map({'No':1,
                                            'Not yet, but still trying':2,
                                            'Just browsing / not trying to accomplish anything specific':3,
                                            'Yes, partly':4,
                                            'Yes, fully':5})
                    ordinal_data.append(acc)
                else:
                    exp = X_train[col].map({'Very poor':1,
                                            'Poor':2,
                                            'Fair':3,
                                            'Good':4,
                                            'Very good':5})
                    ordinal_data.append(exp)

            matrix = pd.concat(ordinal_data,axis=1).values
            return matrix
        else:
            pass


    def fit(self, X_train, y=None):
        return self


class NominalEncoder(BaseEstimator, TransformerMixin):
    """
    Description:
        This class is designed for use as a transformer within an sklearn
        pipeline. The pipeline will call the fit and transform instance methods.

        The transform method will use LabelEncoder() from sklearn to encode
        a nominal variable.
    """
    def __init__(self, cols):
        self.cols = cols


    def transform(self, X_train, y=None):

        if len(self.cols)>1:
            encoded_series = []
            for col in self.cols:
                encoder = LabelEncoder()
                encoder.fit(X_train[col])
                encoded_series.append(pd.Series(encoder.transform(X_train[col])))
            matrix = pd.concat(encoded_series, axis=1).values
            return matrix
        else:
            encoder = LabelEncoder()
            encoder.fit(X_train)
            matrix = encoder.transform(X_train).reshape(-1, 1)
            return matrix


    def fit(self, X_train, y=None):
        return self


class CharLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Description:
        This class is designed for use as a transformer within an sklearn
        pipeline. The pipeline will call the fit and transform instance methods.

        The transform method returns the string len() of of each doc.

    """

    def __init__(self):
        pass


    def transform(self, X_train, y=None):
        #the reshape ensures that the 1D array becomes 2D"""
        matrix = X_train.apply(lambda x: len(x)).values.reshape(-1,1).astype('float')
        return matrix


    def fit(self, X_train, y=None):
        return self

class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Description:
        This class is designed for use as a transformer within an sklearn
        pipeline. The pipeline will call the fit and transform instance methods.

        The transform method returns the datetime of each comment as an ordinal
        cyclic variable, with each datetime component, e.g. the hour of the day,
        represnted by a trigonometric (x,y) pair of coordinates on a unit cirle.
    """

    def __init__(self):
        pass

    @staticmethod
    def angular_day_of_year(unit):
        xday = sin(2*pi*unit/365)
        yday = cos(2*pi*unit/365)
        return xday, yday

    @staticmethod
    def angular_hour(unit):
        t = str(unit)
        (h, m, s) = t.split(':')
        result = int(h) + int(m) / 60
        xhr = sin(2*pi*result/24)
        yhr = cos(2*pi*result/24)
        return xhr, yhr

    @staticmethod
    def angular_month(unit):
        xmonth = sin(2*pi*unit/12)
        ymonth = cos(2*pi*unit/12)
        return xmonth, ymonth

    @staticmethod
    def angular_weekday(unit):
        xweekday = sin(2*pi*unit/7)
        yweekday = cos(2*pi*unit/7)
        return xweekday, yweekday

    def transform(self, X_train, y=None):
        hours = X_train.dt.time.apply(self.angular_hour).apply(pd.Series)
        hours.columns = ['xhr','yhr']
        weekdays = X_train.dt.weekday.apply(self.angular_weekday).apply(pd.Series)
        weekdays.columns = ['xweekday','yweekday']
        month = X_train.dt.month.apply(self.angular_month).apply(pd.Series)
        month.columns = ['xmonth','ymonth']
        day_of_year = X_train.dt.dayofyear.apply(self.angular_day_of_year).apply(pd.Series)
        day_of_year.columns = ['xday','yday']
        matrix = pd.concat([hours,month,weekdays,day_of_year],axis=1).values
        return matrix

    def fit(self, X_train, y=None):
        return self


############################################################################
#Learn/Instantiate Word Embeddings
############################################################################
class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Description:
        This class is designed for use as a transformer within an sklearn
        pipeline. The pipeline will call the fit and transform methods.

        The class attributes instantiate the glove, fastext, and word2vec word
        embedding models. These models are dictionaries mapping unique words
        from the entire corpus (e.g. all of the Normalized Value Comments) to
        vectors of shape [300,]. You can choose the model at __init__,
        which allows you to use GridSearchCV to try each word embedding model
        as well as the usual tf-idf.

    """
    print("_"*80)
    print("Checking for word embeddings...")
    wv_path = os.path.join("wv","wv.bin")
    if not os.path.exists("wv"):
        print("Word embeddings don't exist. Downloading now...")
        os.mkdir("wv")
        # see other options here:
        # https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list.json
        word_vectors = api.load("glove-twitter-200")
        word_vectors.save_word2vec_format(wv_path, binary=True)
    else:
        print("Word embeddings already exist. Loading now...")
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(wv_path, binary=True)
    print("Done loading working embeddings.")
    print("_"*80)


    def __init__(self, vectorizer="ft", word_vectors=word_vectors):
        """
        Description:
            Create an instance of the class with the chosen model.

        Arguments:
            vectorizer:  a str representing the model you'd like to use to
                         vectorize the text. Possible values:
                            'ft': the FastText model
                            'tf_idf':  TfidfVectorizer()
        """
        self.vectorizer = vectorizer
        self.word_vectors = word_vectors


    def fit(self, X_train, y=None):
        """
        Description:
            When this method is called by the sklearn pipeline, it either creates the
            tf_idf scores for the words or the fasttext word embeddings.
            These will be used by transform as weights when aggregating the
            vector representations of each word at the doc level.

        """

        if self.vectorizer == "ft":
            # Creating the model

            self.model = {w: vec for w, vec in zip(self.word_vectors.wv.vocab.keys(),
                                                   self.word_vectors.wv.vectors)}
            self.dim = len(next(iter(self.model.values())))

            # pass callable to analyzer to extract the sequence of features out of the raw, unprocessed comment.
            tfidf = TfidfVectorizer(analyzer=lambda x: x)
            tfidf.fit(X_train)
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(lambda: max_idf,
                                           [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
            return self
        else:
            self.tf_idf = TfidfVectorizer(min_df=3,
                                         max_features=None,
                                         analyzer='word',
                                         ngram_range=(1,2),
                                         use_idf=1,
                                         smooth_idf=1,
                                         sublinear_tf=1)
            self.tf_idf.fit(X_train)
            self.word2weight = None
            return self


    def transform(self, X_train):
        if self.vectorizer != "tf_idf":
            comments  = [[word for word in comment.split(" ")] for comment in X_train]
            embeddings =  np.array([np.mean([self.model[w] * self.word2weight[w]
                                             for w in words if w in self.model] or
                            [np.zeros(self.dim)], axis=0) for words in comments]).astype('float')
            return embeddings
        else:
            return self.tf_idf.transform(X_train).toarray()
