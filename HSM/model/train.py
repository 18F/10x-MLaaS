import dill as pickle
import pandas as pd
import warnings
import os
import sys
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
#in order to use SMOTE, you've got to import Pipeline from imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from model.transformers import ColumnExtractor, OrdinalTransformer, CharLengthExtractor, DateTransformer, TfidfEmbeddingVectorizer
from model.text_utils import NormalizeText

warnings.filterwarnings('ignore')


class TrainClassifer():
    """
    Description:
        This class will train a model depending on the comment_question you
        pass into it during initialization.
    Attributes:
        comment_question (str): choose one of the following as a shorthand
                                for the page-level survey comment questions:
                                Value, Other Purpose of Visit, Purpose of Visit,
                                 Unable to Complete Purpose Reason.
        metric (str): the classifier scoring metric to use. Choose from:
        accuracy, roc_auc, precision, fbeta, or recall. Note that for fbeta,
        beta = 2.
    """

    def __init__(self,comment_question,metric='roc_auc'):
        self.comment_question = comment_question
        self.metric = metric


    def prepare_train(self):

        ordinal_questions = ["Experience Rating",
                             "Likely to Return",
                             "Likely to Recommend",
                             "Able to Accomplish"
                             ]
        labeled_data_path = os.path.join('model','training_data','train.csv')
        labeled_data_df = pd.read_csv(labeled_data_path,encoding='latin1')
        labeled_data_df[self.comment_question] = labeled_data_df[self.comment_question].astype(str)
        print("\tNormalizing the text...")
        nt = NormalizeText()
        normalized_text = nt.transform(labeled_data_df[self.comment_question])
        labeled_data_df['Normalized '+self.comment_question] = normalized_text
        print("\tDone normalizing the text.")
        print("_"*80)
        return labeled_data_df


    def grid_search(self,data):
        """
            Description:
                Given the survey dataset where a comment column has already been
                normalized, split the data into training and test subsets; apply
                custom transformers for feature extraction; and gridsearch
                the following models:  ExtraTreesClassifier(),
                GradientBoostingClassifier(), SGDClassifier(),
                LogisticRegression(), and LinearSVC().

            Parameters:
                data:  the pandas dataframe containing the survey responses as
                       well as two new columns containing binary encoding of
                       spam vs ham and the normalized comment.
                normalized_comment_col:  the str name of the normalized comment
                                         column in data.
                comment_col:  the str name of the comment column in data.
                label_col:  the str name of the column containing the class
                            labels.
                date_col:  the str name of the datetime column in data.
                scoring_metric:  The scoring metric to be used when refitting
                                the models. 'roc_auc' by default.
        """
        normalized_comment_col = "Normalized " + self.comment_question
        comment_col = self.comment_question
        label_col = self.comment_question + " Spam"
        date_col = "EndDate"
        scoring_metric = self.metric
        ordinal_questions = ["Experience Rating",
                             "Likely to Return",
                             "Likely to Recommend",
                             "Able to Accomplish"
                             ]
        X = data.drop(label_col, axis =1)
        y = data[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.1,
                                                            random_state=123)
        classifiers = [SGDClassifier(),
                       ExtraTreesClassifier(),
                       GradientBoostingClassifier(),
                       LogisticRegression(),
                       LinearSVC()]
        scoring = {'accuracy': metrics.make_scorer(metrics.accuracy_score),
                   'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
                   'precision': metrics.make_scorer(metrics.average_precision_score),
                   'fbeta':metrics.make_scorer(metrics.fbeta_score,beta=2),
                   'recall':metrics.make_scorer(metrics.recall_score)}
        results = {k:None for k in [clf.__class__.__name__ for clf in classifiers]}
        for clf in classifiers:
            print('=' * 80)
            clf_name = clf.__class__.__name__
            print("Training {}...".format(clf_name))
            print('=' * 80)
            pipe = Pipeline([
                             ('features', FeatureUnion([
                                         ('word2vec', Pipeline([
                                                           ('extractor', ColumnExtractor(cols=[normalized_comment_col],
                                                                                      dtype='str')),
                                                           ('vectorizer', TfidfEmbeddingVectorizer())
                                         ])),
                                         ('comment_length', Pipeline([
                                                           ('extractor',ColumnExtractor(cols=[comment_col],
                                                                                        dtype='str')),
                                                           ('num_chars', CharLengthExtractor())
                                         ])),
                                         ('ordinal', Pipeline([
                                                            ('extract', ColumnExtractor(cols = ordinal_questions,
                                                                                        dtype='str')),
                                                            ('ordinal_enc', OrdinalTransformer(cols = ordinal_questions))
                                         ])),
                                         ('datetime', Pipeline([
                                                           ('extract', ColumnExtractor(cols = [date_col],
                                                                                       dtype='datetime')),
                                                           ('date_transform', DateTransformer())
                                         ])),
                             ])),
                            ('scaler', StandardScaler(with_mean=False)),
                            ('upsample', SMOTE()),
                            ('select', SelectPercentile(f_classif)),
                            ('clf', clf)])

            if "Extra" in clf_name:
                param_grid = [
                                {'clf__max_depth': [15],
                                 'clf__max_features': [None],
                                 'clf__min_samples_leaf': [2],
                                 'clf__min_samples_split': [2],
                                 'clf__n_estimators': [1200],
                                 'features__word2vec__vectorizer__vectorizer': ['ft'],
                                 'select': [None],
                                 'upsample': [None]}
                               ]
            elif 'Gradient' in clf_name:
                param_grid = [
                                {
                                  'features__word2vec__vectorizer__vectorizer':['ft'],
                                  'upsample':[None],
                                  'select': [None],
                                  'clf__n_estimators':[1200],
                                  'clf__learning_rate':[0.1],
                                  'clf__max_depth':[3],
                                  'clf__max_features':['sqrt'],
                                }
                               ]
            elif "Linear" in clf_name:
                param_grid = [
                                {
                                  'features__word2vec__vectorizer__vectorizer':['ft'],
                                  'upsample':[None],
                                  'select': [None],
                                  'clf__C':[.001],
                                  'clf__penalty':['l2'],
                                  'clf__loss':['squared_hinge'],
                                  'clf__class_weight':['balanced']
                                }
                            ]
            elif "Logistic" in clf_name:
                param_grid = [
                                {
                                  'features__word2vec__vectorizer__vectorizer':['ft'],
                                  'upsample':[None],
                                  'select': [None],
                                  'clf__penalty':['l2'],
                                  'clf__C':[.01],
                                  'clf__class_weight':['balanced']
                                }
                            ]
            elif "SGD" in clf_name:
                param_grid = [
                                {
                                  'features__word2vec__vectorizer__vectorizer':['ft'],
                                  'upsample':[None],
                                  'select': [None],
                                  'clf__penalty':['l2'],
                                  'clf__loss':['modified_huber'],
                                  'clf__alpha':[1e0],
                                  'clf__class_weight':[None]
                                }
                            ]
            gs = GridSearchCV(pipe, param_grid = param_grid, scoring = scoring,
                              refit = scoring_metric,
                              n_jobs = -1, verbose = True, cv = 5,
                              return_train_score = False)
            gs.fit(X_train, y_train)
            best_score = gs.best_score_
            best_params = gs.best_params_
            y_pred = gs.predict(X_test)
            #get the col number of the positive class (i.e. spam)
            positive_class_col = list(gs.classes_).index(1)
            try:
                y_score = gs.predict_proba(X_test)[:,positive_class_col]
            except AttributeError:
                y_score = gs.decision_function(X_test)
            average_precision = metrics.average_precision_score(y_test, y_score)
            acc = metrics.accuracy_score(y_test,y_pred)
            roc_auc = metrics.roc_auc_score(y_test, y_pred)
            precisions, recalls, _ = metrics.precision_recall_curve(y_test, y_score)
            auc = metrics.auc(recalls, precisions)
            # beta > 1 favors recall
            fbeta = metrics.fbeta_score(y_test,y_pred,beta=2)
            recall = metrics.recall_score(y_test,y_pred)
            best_estimator = gs.best_estimator_

            results[clf_name] = (precisions, recall, average_precision,
                                 acc, roc_auc, auc, fbeta, recalls, best_params,
                                 best_score, best_estimator)

            print("Best score on training data:  {0:.2f}".format(gs.best_score_))
            print("\tRecall on test data:  {0:.2f}".format(recall))
            print("\tAccuracy on test data:  {0:.2f}".format(acc))
            print("\tROC-AUC on test data:  {0:.2f}".format(roc_auc))
            print("\tFbeta on test data:  {0:.2f}".format(fbeta))
            print("\tAverage Precision on test data:  {0:.2f}".format(average_precision))
            print("\tPrecision-Recall AUC on test data:  {0:.2f}".format(auc))
            print(f"\tBest estimator:{best_estimator}")
            print(f"\tBest params:{best_params}")

        return results


    def pickle_model(self,results):
        metric=self.metric
        print("Pickling the best model...")
        #get the classifier scores depending on metric
        if metric == 'accuracy':
            best_scores = [results[k][3] for k in results.keys()]
        elif metric == 'precision':
            best_scores = [results[k][2] for k in results.keys()]
        elif metric == 'fbeta':
            best_scores = [results[k][6] for k in results.keys()]
        elif metric == 'roc_auc':
            best_scores = [results[k][4] for k in results.keys()]
        elif metric == 'recall':
            best_scores = [results[k][1] for k in results.keys()]
        #find the index of the max score, which corresponds to the classifier
        index_max = max(range(len(best_scores)), key=best_scores.__getitem__)
        best_model_name = list(results.keys())[index_max]
        #look up the best_estimator
        best_model = results[best_model_name][-1]
        pickle_dir = os.path.join(os.getcwd(),'model','best_estimators')
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        pickle_file = os.path.join(pickle_dir,
                                   f'{self.comment_question}_{metric}_best_estimator.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(best_model, f)
        print("Done pickling the best model.")


    def show_clf_report(self,results):
        metric=self.metric
        # TODO: display different reports depending on metric
        for k in results:
            precisions = results[k][0]
            average_precision = results[k][2]
            recalls = results[k][7]

            plt.figure(figsize=(8,6))
            plt.step(recalls, precisions, color='b', alpha=0.2,
                 where='post')
            plt.fill_between(recalls, precisions, step='post', alpha=0.2,
                             color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('{} --- 2-class Precision-Recall curve'.format(k))
            plt.show()
