import os
import time
import pandas as pd
from argparse import ArgumentParser
from utils import config, qualtrics, validate, db, db_utils
from model import predict, train


def get_survey_data(session, excel_filename=None):
    '''
    Implements the Qualtrics class to get most recent qualtrics survey data from their API.

    Parameters:
        session: a sqlalchemy session object
        excel_filename: The filename of the Excel file that has the input

    Returns:
        df (pandas DataFrame): a dataframe of the survey data
    '''
    if excel_filename:
        print(f'******** Getting data from Excel Spreadsheet at location {excel_filename} ********')
        input_path = os.path.join(config.INPUT_DIR, excel_filename)
        df = pd.read_excel(input_path)

    else:  # QUALTRICS
        print(f'******** Reading data from Qualtrics ********')
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

    df['prediction'] = df['ResponseID'].map(id_pred_map)
    df['validated prediction'] = df['ResponseID'].map(validated_id_pred_map)
    db_utils.insert_data(df, session)


def retrain_model(session):
    '''
    Retrain the model by getting all data from database to do so.  Called train to train the model and save
    it to the designated place.

    Parameters:
        session: a database session that will be passed in to access the database data
    '''

    df_comment_spam = db_utils.get_data(session, filter_feature='Comments Concatenated', validation='SPAM')

    train.main(df_comment_spam)


def main(survey_name="Site-Wide Survey English", model_description="model_sw.pkl", excel_filename=None):
    '''
    Create db if it doesn't exist; fetch survey data from Qualtrics; make predictions; provide the user
    with a chance to validate the predictions in a spreadsheet; and insert data into db.
    '''

    db_utils.create_postgres_db()
    db.dal.connect()
    session = db.dal.Session()
    df = get_survey_data(session, excel_filename)
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
    parser.add_argument("-i", "--input", default=None,
                        help="specify input excel file name else Qualtrics API is being used to get data, file is "
                             "expected to be saved at 10x-MLaaS/HSM/model/inputs folder")

    args = parser.parse_args()

    main(survey_name=args.survey_name, model_description=args.model, excel_filename=args.input)
