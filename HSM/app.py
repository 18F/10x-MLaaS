import qualtrics
import update_db
import os
from model import predict, train
import sys
import time

# Get the data from qualtrics
print("-"*80)
qa = qualtrics.QualtricsApi()
qa.download_responses()
qa.update_db()

model_path = os.path.join(os.getcwd(),
                          'model',
                          'best_estimators',
                          'model_sw.pkl')

if os.path.exists(model_path):
    print("A trained model already exists, so let's use it!")
    db_path = os.path.join(os.getcwd(),'db','db.csv')
    nd = predict.ClassifyNewData(db_path)
    nd.get_new_data()
    nd.predict()
    print("Done making predictions. You can find the results in ClassificationResults.xlsx")
    print('-'*80)

    print("Take a moment to review the predictions. Change those that you disagree \
           with. When you're done, save and exit the spreadsheet. Then return to \
           this terminal shell.")

    time.sleep(10)
    user_input = ''
    while user_input != 'y':
        user_input = str(input("If you've finished reviewing the predictions, enter 'y': "))

    ud = update_db.UpdateDb()
    ud.update_db()
    print("Done updating the database with your reviewed predictions!")
    sys.exit(0)
else:
    print("A trained model doesn't already exists, so let's train one now!")
    tc = train.TrainClassifer()
    train_df = tc.prepare_train()
    results = tc.randomized_grid_search(train_df)

print('-'*80)
print("Making predictions on new data using the trained model...")
db_path = os.path.join(os.getcwd(),'db','db.csv')
nd = predict.ClassifyNewData(db_path)
nd.get_new_data()
nd.predict()
print("Done making predictions. You can find the results in ClassificationResults.xlsx")
print('-'*80)

print("Take a moment to review the predictions. Change those that you disagree \
       with. When you're done, save and exit the spreadsheet. Then return to \
       this terminal shell.")

time.sleep(10)

user_input = ''
while user_input != 'y':
    user_input = str(input("If you've finished reviewing the predictions, enter 'y': "))

ud = update_db.UpdateDb()
ud.update_db()
print("Done updating the database with your reviewed predictions!")
