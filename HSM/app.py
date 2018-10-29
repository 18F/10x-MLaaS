import qualtrics
import validate
import os
import pandas as pd
from model import predict, train
import db
import sys
import time

def get_survey_data(session):
    last_response_id = db.fetch_last_RespondentID(session)
    qa = qualtrics.QualtricsApi(last_response_id)
    qa.download_responses()
    df = qa.get_data()
    return df

def make_predictions(df):
    mp = predict.MakePredictions(df, survey_type='sw')
    results_path, df, id_pred_map = mp.predict()
    return results_path, df, id_pred_map

def user_prompt():
    print("Done making predictions. You can find the results in ClassificationResults.xlsx")
    print('-'*80)
    print("Take a moment to review the predictions.")
    print("Change those that you disagree with.") 
    print("When you're done, save and exit the spreadsheet. Then return here.")
    time.sleep(10)
    user_input = ''
    while user_input != 'y':
        user_input = str(input("If you've finished reviewing the predictions, enter 'y': "))
    print("Inserting data into database. Hold on...")

def get_validations(results_path):
    v = validate.Validate(results_path)
    validated_id_pred_map = v.get_validations()
    return validated_id_pred_map

def insert_data(df, validated_id_pred_map, id_pred_map, survey_name, model_description, session):
    survey_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Comments_Concatenated']
    respondent_attributes = [x.replace(" ","_") for x in df.columns\
                             if x not in survey_questions]
    df['prediction'] = df['ResponseID'].map(id_pred_map)
    df['validated prediction'] = df['ResponseID'].map(validated_id_pred_map)
    prediction_set = set(df['prediction'])
    validation_set = set(df['validated prediction'])
    db.survey_exists(survey_name, survey_questions, session)
    db.model_exists(model_description, session)
    db.validation_exists(validation_set, session)
    db.prediction_exists(prediction_set, session)
    db.insert_respondents(df, respondent_attributes, session)
    db.insert_responses(df, survey_questions, survey_name, model_description, session)
    
def main(survey_name = "Site-Wide Survey English", model_description = "model_sw.pkl"):
    db.create_postgres_db()
    db.dal.connect()
    session = db.dal.Session()
    df = get_survey_data(session)
    results_path, df, id_pred_map = make_predictions(df)
    user_prompt()
    validated_id_pred_map = get_validations(results_path)
    insert_data(df, validated_id_pred_map, id_pred_map, survey_name, model_description, session)
    session.commit()

if __name__ == '__main__':
    main()
    