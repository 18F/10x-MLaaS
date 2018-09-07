import qualtrics
import os

qa = qualtrics.QualtricsApi()
qa.download_responses()
qa.update_db()

model_path = os.path.join(os.getcwd(),
                          'model',
                          'best_estimators',
                          'Valuetest_roc_auc_best_estimator.pkl')
if os.path.exists(model_path):
    pass
    print("A trained model already exists, so let's use it!")
    # TODO: check if we've hit a threshold for newly labeled data

else:
    pass
    # TODO: train a brand new model
