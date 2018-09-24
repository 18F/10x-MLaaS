# 10x-qualitative-data
Repository for the 10x qualitative data management project. 

## HSM
`app.py` is a command line interface for the Ham-Spam Machine (HSM). Currently, it only fethces and classifies responses to the **site-wide version** of the survey.

Before you can run the script, you'll need the following assets:
 - your Qualtrics credentials placed within the HSM root directory as `HSM/secrets.json`. For example:
 ```
 {
  "apiToken":"yourToken",
  "surveyId":"yourSurveyID"
 }
 ```
 - the training data as `train.csv` placed here: `HSM/model/training_data/train-sw.csv`  

Once you've got those assets (and all of the Python dependencies installed) you're ready to run `python app.py`. Here's what will happen when you do:
 - Download data from the Qualtrics API. If you don't have any responses yet, then you're running this for the first time and it'll download all of the responses to-date, placing them in `db.csv`. If you've already run this once, it'll only download the responses since the last time you executed the script.
 - Trains a ham-spam classifier if you don't already have one (takes about 30 minutes)
 - Uses the classifier to predict on the new data you've just downloaded, writing results to an excel file that you will review
 - Without exiting the scritp, you then review the results in  `HSM/model/results/ClassificationResults.xlsx`
    - When reviewing the results, the prediction is in the `SPAM` column (0 = ham and 1 = spam). Make your changes inplace, overwriting the prediction if you disagree.
 - Once you've been able to review the predictions and save your corrections in `ClassificationResults.xlsx`, tell the script you've done so by following the prompt. The script will write those corrected predictions back into the database (`HSM/db/db.csv`)
 - And then you're done! Wait a week or so to repeat.
 
 In the future, it'll:
 - log performance of the models (based on the ground truth established by the review)
 - log performance of model training iterations
 - only retrain the model if a certain threshold of newly labeled and human-validated samples is met
 - extend to the page-level survey
 - be dockerized
 - use PostgreSQL for the database
 - have an API that exposes all of these back-end operations as well as the database
 - have unit tests and CI
