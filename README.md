# 10x-qualitative-data
Repository for the 10x qualitative data management project. 

## VoC Notebooks
This contains the original Jupyter Notebooks that developed the ham-spam classifier, which we've come to term the Ham-Spam Machine (HSM).

## HSM
This will contain scripts to:
 - get survey responses from the Qualtrics API
 - train a model if you don't already have one pickled
 - predict on new data, writing results to an excel file for review
 - write human-reviewed predictions back to the database (currently a csv)
 - log performance of the models (based on the ground truth established by the review)
 - log performance of model training iterations
 - retrain the model if a certain threshold of newly labeled and human-validated samples is met
