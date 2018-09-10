# 10x-qualitative-data
Repository for the 10x qualitative data management project. 

## VoC Notebooks
This contains the original Jupyter Notebooks that developed the ham-spam classifier, which we've come to term the Ham-Spam Machine (HSM).

## HSM
`app.py` is a command line interface version of the Ham-Spam Machine (HSM). Currently, it only classifies comments in response to the **Please tell us what you value most about this site** question on the **site-wide version** of the survey.

Before you can run the script, you'll need a `secrets.json` with your Qualtrics credentials placed within the root directory. You'll also need the training data (`model/training_data/train.csv`) and a list of profanity (`model/corpora/profanity.csv`). 

Once you've got those things, here's what happens when you run `python app.py`:
 - Gets survey responses from the Qualtrics API using the latest responseId on hand. If you don't have any responses yet, then you're running this for the first time and it'll download all of the responses to-date, placing them in `db.csv`
 - Trains a ham-spam classifier if you don't already have one (takes about 10 minutes)
 - Uses the classifier to predict on the new data you've just downloaded, writing results to an excel file for review
 - Once you've been able to review the predictions and save your corrections, tell the script you've done so and it'll write the corrected predictions back into the database
 
 In the future, it'll:
 - log performance of the models (based on the ground truth established by the review)
 - log performance of model training iterations
 - only retrain the model if a certain threshold of newly labeled and human-validated samples is met
 - extend to the three other page-level comment fields
