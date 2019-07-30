import time
from argparse import ArgumentParser
import pandas as pd
from utils import qualtrics, validate, db, db_utils
from model import predict, train


def get_survey_data(session):
    '''
    Implements the Qualtrics class to get most recent qualtrics survey data from their API.

    Parameters:
        session: a sqlalchemy session object

    Returns:
        df (pandas DataFrame): a dataframe of the survey data
    '''

    last_response_id = db_utils.fetch_last_RespondentID(session)
    qa = qualtrics.QualtricsApi(last_response_id)
    qa.download_responses()
    df = qa.get_data()

    return df


def make_predictions(df):
    '''
    Given a dataframe of survey data, parse the comments, feed them to model, and make predictions

    Parameters:
        df (pandas DataFrame): a dataframe of the survey data, as returned by get_survey_data()

    Returns:
        results_path (str): abs path to the results, ClassificationResults.xlsx
        df (pandas DataFrame): a dataframe of the survey data, with new columns for the predictions and decision
        boundary
        id_pred_map (dict): a dict mapping Qualtrics ResponseIDs to the SPAM predition (i.e. 0 (ham) or 1 (spam))
    '''

    mp = predict.MakePredictions(df, survey_type='sw')
    results_path, df, id_pred_map, outfile = mp.predict()

    return results_path, df, id_pred_map, outfile


def user_prompt(outfile):
    print("Done making predictions. You can find the results in {}".format(outfile))
    print('-'*80)
    print("Take a moment to review the predictions.")
    print("Change those that you disagree with.")
    print("When you're done, save and exit the spreadsheet. Then return here.")
    time.sleep(10)
    user_input = ''
    while user_input != 'y':
        user_input = str(input("If you've finished reviewing the predictions, enter 'y': "))
    print("Inserting data into database. It may take a while.  Hold on...")


def get_validations(results_path):
    '''
    After the user has had time to review the predictions and make/save corrections, read in the file and get
    validations

    Parameters:
        results_path (str): abs path to the results, ClassificationResults.xlsx

    Returns:
        validated_id_pred_map (dict): a dict mapping Qualtrics ResponseIDs to the user-validated SPAM predictions
    '''
    v = validate.Validate(results_path)
    validated_id_pred_map = v.get_validations()

    return validated_id_pred_map


def insert_data(df, validated_id_pred_map, id_pred_map, survey_name, model_description, session):
    '''
    Void function to insert data into postgres database, as configured in config.py

    Parameters:
        df (pandas DataFrame): a dataframe of the survey data, as returned by get_predictions()
        validated_id_pred_map (dict): a dict mapping Qualtrics ResponseIDs to the user-validated SPAM predictions
        id_pred_map (dict): a dict mapping Qualtrics ResponseIDs to the SPAM prediction (i.e. 0 (ham) or 1 (spam))
        survey_name (str): name of the survey we're dealing with
        model_description (str): file name for the model (e.g. model_sw.pkl)
        session: a sqlalchemy session object
    '''

    survey_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Comments_Concatenated']
    respondent_attributes = [x.replace(" ", "_") for x in df.columns
                             if x not in survey_questions]
    df['prediction'] = df['ResponseID'].map(id_pred_map)
    df['validated prediction'] = df['ResponseID'].map(validated_id_pred_map)
    prediction_set = set(df['prediction'])
    validation_set = set(df['validated prediction'])
    db_utils.survey_exists(survey_name, survey_questions, session)
    db_utils.model_exists(model_description, session)
    db_utils.validation_exists(validation_set, session)
    db_utils.prediction_exists(prediction_set, session)
    db_utils.insert_respondents(df, respondent_attributes, session)
    db_utils.insert_responses(df, survey_questions, survey_name, model_description, session)


def retrain_model(session):
    # Find the id of the comments_concatentated row
    question_id = session.query(db.Question).filter(db.Question.text == "Comments_Concatenated").one().id
    comment_spam = session.query(db.Response.text, db.Validation.validation) \
                          .filter(db.Response.question_id == question_id) \
                          .filter(db.Response.respondent_id == db.Respondent.id) \
                          .filter(db.Validation.id == db.Response.validation_id).all()
    df_comment_spam = pd.DataFrame(comment_spam, columns=['Comments Concatenated', 'SPAM'])
    train.main(df_comment_spam)


def main(survey_name="Site-Wide Survey English", model_description="model_sw.pkl"):
    '''
    Create db if it doesn't exist; fetch survey data from Qualtrics; make predictions; provide the user
    with a chance to validate the predictions in a spreadsheet; and insert data into db.
    '''

    db_utils.create_postgres_db()
    db.dal.connect()
    session = db.dal.Session()
    df = get_survey_data(session)
    results_path, df, id_pred_map, outfile = make_predictions(df)
    user_prompt(outfile)
    validated_id_pred_map = get_validations(results_path)

    insert_data(df, validated_id_pred_map, id_pred_map, survey_name, model_description, session)
    session.commit()

    retrain_model(session)

    print("DONE!")


if __name__ == '__main__':

    program_desc = '''This application will get survey data from Qualtrics and make prediction on the data.
                      It will then retrain the model based on the validated data.'''

    parser = ArgumentParser(description=program_desc)
    parser.add_argument("-s", "--survey_name", dest="survey_name",
                        help="specify survey name to use", default="Site-Wide Survey English")
    parser.add_argument("-m", "--model",
                        default="model_sw.pkl",
                        help="specify model file name")

    args = parser.parse_args()

    main(survey_name=args.survey_name, model_description=args.model)
