import dill as pickle
import pandas as pd
import warnings
import os
import sys
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
#in order to use SMOTE, you've got to import Pipeline from imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from transformers import NormalizeText, ColumnExtractor, OrdinalTransformer, CharLengthExtractor, DateTransformer, TfidfEmbeddingVectorizer

warnings.filterwarnings('ignore')

##############################################################################
# Load the training data
##############################################################################
comment_questions = ['Value',
                    "Other Purpose of Visit",
                    "Purpose of Visit",
                    "Unable to Complete Purpose Reason"
                     ]
comment_question = comment_questions[0]
ordinal_questions = ["Experience Rating",
                     "Likely to Return",
                     "Likely to Recommend",
                     "Able to Accomplish"
                     ]

labeled_data_path = os.path.join('training_data','train.csv')
labeled_data_df = pd.read_csv(labeled_data_path,encoding='latin1')
labeled_data_df[comment_question] = labeled_data_df[comment_question].astype(str)
##############################################################################
# Normalize the text
##############################################################################
print("_"*80)
print("Normalizing the text...")
nt = NormalizeText()
normalized_text = nt.transform(labeled_data_df[comment_question])
labeled_data_df['Normalized '+comment_question] = normalized_text
print("Done normalizing the text.")
print("_"*80)
##############################################################################
# Grid Search
##############################################################################
def grid_search(data=labeled_data_df,
                normalized_comment_col="Normalized "+comment_question,
                comment_col=comment_question,
                ordinal_questions = ordinal_questions,
                label_col = "Value Spam",
                date_col = "EndDate",
                scoring_metric = 'roc_auc'):
    """
        Description:
            Given the survey dataset where a comment column has already been normalized, split the data into
            training and test subsets; apply custom transformers for feature extraction; and gridsearch
            the following tree-based and linear models:  RandomForestClassifier(), ExtraTreesClassifier(),
            AdaBoostClassifier(), GradientBoostingClassifier(), SGDClassifier(), LogisticRegression(), and
            LinearSVC().

        Parameters:
            data:  the pandas dataframe containing the survey responses as well as two new columns containing
                   binary encoding of spam vs ham and the normalized comment.
            normalized_comment_col:  the str name of the normalized comment column in data.
            comment_col:  the str name of the comment column in data.
            label_col:  the str name of the column containing the class labels.
            date_col:  the str name of the datetime column in data.
            stratify_shuffle_split:  Boolean to implement stratified shuffle splitting instead of a random
                                     train-test-split.
            scoring_metric:  The scoring metric to be used when refitting the models. 'accuracy' by default,
                             but can also be set to 'roc_auc'.


    """

    X = data.drop(label_col, axis =1)
    y = data[label_col]


    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        random_state=123)


    classifiers = [RandomForestClassifier(),
                  ExtraTreesClassifier(),
                  AdaBoostClassifier(),
                  GradientBoostingClassifier(),
                  SGDClassifier(),
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


        if 'Random' in clf_name:
            param_grid = [
                            {'clf__max_depth': [5,20,None],
                             'clf__max_features': ['sqrt',None],
                             'clf__min_samples_leaf': [2,5,10],
                             'clf__min_samples_split': [2,5,10],
                             'clf__n_estimators': [300,500,800,1200],
                             'features__word2vec__vectorizer__vectorizer': ['ft','tf_idf'],
                             'select': [None],
                             'upsample': [None]}
                        ]

        elif "Extra" in clf_name:
            param_grid = [
                            {'clf__max_depth': [5,20,None],
                             'clf__max_features': ['sqrt',None],
                             'clf__min_samples_leaf': [2,50,10],
                             'clf__min_samples_split': [2,5],
                             'clf__n_estimators': [500,800,1200],
                             'features__word2vec__vectorizer__vectorizer': ['ft','tf_idf'],
                             'select': [None],
                             'upsample': [None]}
                        ]


        elif 'Gradient' in clf_name:
            param_grid = [
                            {
                              'features__word2vec__vectorizer__vectorizer':['ft','tf_idf'],
                              'upsample':[None],#,SMOTE(kind='svm')
                              'select': [None],#SelectPercentile(f_classif, percentile=50)
                              'clf__n_estimators':[1200],
                              'clf__learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                              'clf__max_depth':[3,5,8,10,15,30],
                              'clf__max_features':['sqrt'],
                            }
                        ]

        elif "AdaBoost" in clf_name:
            param_grid = [

                            {
                              'features__word2vec__vectorizer__vectorizer':['ft','tf_idf'],
                              'upsample':[None],
                              'select': [None],
                              'clf__n_estimators':[400,800,1200],
                              'clf__learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
                            }
                        ]

        elif "Linear" in clf_name:
            param_grid = [
                            {
                              'features__word2vec__vectorizer__vectorizer':['ft','tf_idf'],
                              'upsample':[None],
                              'select': [None],
                              'clf__C':[.001,.01,.1,1.0,10,100],
                              'clf__penalty':['l1','l2'],
                              'clf__loss':['hinge','squared_hinge'],
                              'clf__class_weight':[None]
                            }
                        ]

        elif "Logistic" in clf_name:
            param_grid = [
                            {
                              'features__word2vec__vectorizer__vectorizer':['ft','tf_idf'],
                              'upsample':[None],
                              'select': [None],
                              'clf__penalty':['l2','l1'],
                              'clf__C':[100,10,1.0,.1,.01,.001]
                            }
                        ]
        elif "SGD" in clf_name:
            param_grid = [
                            {
                              'features__word2vec__vectorizer__vectorizer':['ft','tf_idf'],
                              'upsample':[None],
                              'select': [None],
                              'clf__penalty':['l2','l1'],
                              'clf__alpha':[1.0,.1,.01]
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

    return results

print("\nBeginning grid search:\n")
results = grid_search()

def pickle_model(results,metric='roc_auc'):
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

    comment_abbreviation = comment_question.split(" ")[0]+"test"
    pickle_dir = os.path.join(os.getcwd(),'best_estimators')
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    pickle_file = os.path.join(pickle_dir,
                               f'{comment_abbreviation}_{metric}_best_estimator.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(best_model, f)
    print("Done pickling the best model.")

pickle_model(results)

def show_clf_report(results,metric='roc_auc'):
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
