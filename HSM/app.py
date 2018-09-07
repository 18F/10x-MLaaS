import qualtrics
import os
from model import predict, train

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
    db_path = os.path.join(os.getcwd(),'db','db.csv')
    nd = predict.ClassifyNewData(db_path,'Value')
    nd.get_new_data()
    nd.predict()
    print("Done making predictions. You can find the results in ClassificationResults.xlsx")

else:
    pass
    # TODO: train a brand new model
