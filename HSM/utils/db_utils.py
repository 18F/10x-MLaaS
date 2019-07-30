import pandas as pd
from utils.config import SQLALCHEMY_URI
from sqlalchemy_utils import database_exists, create_database
from utils.db import (
    Survey,
    Question,
    SurveyQuestion,
    Respondent,
    Response,
    Model,
    Version,
    Prediction,
    Validation,
    VersionPrediction
)
from sqlalchemy import desc, func,  create_engine


def create_postgres_db():
    connection_string = SQLALCHEMY_URI
    engine = create_engine(connection_string, echo=False)
    if not database_exists(engine.url):
        create_database(engine.url)


def survey_exists(survey_name, survey_questions, session):
    survey_names = []
    for row in session.query(Survey.name).all():
        survey_names.append(row.name)

    if survey_name in survey_names:
        pass
    else:
        site_wide = Survey(name="Site-Wide Survey English")
        for q in survey_questions:
            sq = SurveyQuestion()
            sq.question = Question(text=q)
            site_wide.questions.append(sq)
        session.add(site_wide)


def model_exists(model_description, session):
    model_description = model_description.replace(".pkl", "")
    version_description = model_description.rsplit('_', 1)[-1]
    model_description = model_description.replace("_" + version_description, "")
    version = Version(description=version_description)
    model_descriptions = []
    # TODO: Why are we doing this instead of querying the database directly for the row.description match?)
    for row in session.query(Model.description).all():
        model_descriptions.append(row.description)
    if model_description in model_descriptions:
        pass
    else:
        model = Model(description=model_description)
        model.versions.append(version)
        session.add(model)


def validation_exists(validation_set, session):
    validations = []
    for row in session.query(Validation.validation).all():
        validations.append(row.validation)
    diff = validation_set.difference(set(validations))
    if len(diff) > 0:
        validations_to_insert = []
        for v in diff:
            validation = Validation(validation=int(v))
            validations_to_insert.append(validation)
        session.add_all(validations_to_insert)


def prediction_exists(prediction_set, session):
    predictions = []
    for row in session.query(Prediction.prediction).all():
        predictions.append(row.prediction)
    diff = prediction_set.difference(set(predictions))
    if len(diff) > 0:
        predictions_to_insert = []
        for v in diff:
            prediction = Prediction(prediction=int(v))
            predictions_to_insert.append(prediction)
        session.add_all(predictions_to_insert)


def insert_respondents(df, respondent_attributes, session):
    # Get the number of rows here
    for i in range(df.shape[0]):
        if (i % 1000) == 0:
            session.commit()
        data = df.iloc[i][respondent_attributes]
        respondent_data = {k: v for k, v in zip(respondent_attributes, data)}
        respondent = Respondent(**respondent_data)
        session.add(respondent)


def insert_responses(df, survey_questions, survey_name, model_description, session):

    def fetch_survey_id(survey_name, session):
        survey_id = session.query(Survey.id).filter(Survey.name == survey_name).first().id
        return survey_id

    def fetch_question_id(question_name, session):
        question_id = session.query(Question.id).filter(Question.text == question_name).first().id
        return question_id

    def fetch_respondent_id(qualtrics_response_id, session):
        respondent_id = session.query(Respondent.id).filter(Respondent.ResponseID == qualtrics_response_id).first().id
        return respondent_id

    def fetch_validation_id(validation, session):
        validation_id = session.query(Validation.id).filter(Validation.validation == validation).first().id
        return validation_id

    def fetch_model_version_ids(model_description, session):
        model_description = model_description.replace(".pkl", "")
        version_description = model_description.rsplit('_', 1)[-1]
        model_description = model_description.replace("_" + version_description, "")
        model_id = session.query(Model.id).filter(Model.description == model_description).first().id
        version_id = session.query(Version.id).filter(Version.description == version_description).first().id

        return model_id, version_id

    def fetch_prediction_id(prediction, session):
        prediction_id = session.query(Prediction.id).filter(Prediction.prediction == prediction).first().id

        return prediction_id

    survey_id = fetch_survey_id(survey_name, session)
    model_id, version_id = fetch_model_version_ids(model_description, session)

    for i in range(df.shape[0]):
        if (i % 250) == 0:
            # Commit in batches.
            session.commit()
            session.flush()
        data = df.iloc[i][survey_questions+['ResponseID', 'prediction', 'validated prediction']]
        respondent_id = fetch_respondent_id(data['ResponseID'], session)
        validation = int(data['validated prediction'])
        validation_id = fetch_validation_id(validation, session)
        pred = int(data['prediction'])
        prediction_id = fetch_prediction_id(pred, session)
        # prediction = session.query(Prediction).get(prediction_id)
        val = session.query(Validation).get(validation_id)
        for q in survey_questions:
            question_id = fetch_question_id(q, session)
            response = Response(survey_id=survey_id,
                                question_id=question_id,
                                respondent_id=respondent_id,
                                text=data[q])

            val.responses.append(response)
            session.add(response)
            # By flushing we can get the response ID
            session.flush()
            response_id = response.id
            version_prediction = VersionPrediction()
            version_prediction.model_id = model_id
            version_prediction.version_id = version_id
            version_prediction.prediction_id = prediction_id
            version_prediction.response_id = response_id
            # version_prediction.prediction = prediction
            session.add(version_prediction)
        session.commit()


def fetch_last_RespondentID(session):
    '''
    Fetch the last RespondentID from the database to use with the Qualtrics API

    Parameters:
        session: an instance of a sqlalchemy session object created by DataAccessLayer

    Returns:
        last_response_id (str): the RespondentID of the last survey response
    '''
    try:
        query_response = session.query(Respondent).order_by(desc('EndDate')).first()
        last_response_id = query_response.ResponseID
    except AttributeError:
        last_response_id = None

    return last_response_id


def count_table_rows(table, session):
    rows = session.query(func.count(table.id)).scalar()

    return rows


def fetch_concatenated_comments(session):

    def fetch_question_id(question_name, session):
        question_id = session.query(Question.id).filter(Question.text == question_name).first().id
        return question_id

    question_id = fetch_question_id("Comments_Concatenated", session)
    rows = session.query(Response).filter_by(question_id=question_id).all()
    responses = []
    for row in rows:
        if row.text:
            responses.append(row.text)

    return responses


def prep_test_db(session):
    '''
    Create test db and dummy data for testing
    '''
    survey_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Comments_Concatenated']
    respondent_attributes = ['Browser_Metadata_Q_1_TEXT', 'Browser_Metadata_Q_2_TEXT',
                             'Browser_Metadata_Q_3_TEXT', 'Browser_Metadata_Q_4_TEXT',
                             'Browser_Metadata_Q_5_TEXT', 'Browser_Metadata_Q_6_TEXT',
                             'Browser_Metadata_Q_7_TEXT', 'CP_URL', 'Country', 'DeviceType',
                             'EndDate', 'ExternalDataReference', 'Finished', 'History', 'IPAddress',
                             'LocationAccuracy', 'LocationLatitude', 'LocationLongitude', 'PR_URL',
                             'RecipientEmail', 'RecipientFirstName', 'RecipientLastName', 'Referer',
                             'ResponseID', 'ResponseSet', 'SPAM', 'Site_Referrer', 'StartDate', 'State',
                             'Status', 'TVPC', 'UPVC', 'UserAgent', 'Welcome_Text', 'pageType',
                             'Comments_Concatenated']
    pred_cols = ['prediction', 'validated prediction']
    cols = survey_questions+respondent_attributes+pred_cols
    d = {k: ['abc', '123'] for k in cols}
    df = pd.DataFrame(d)
    # set up survey and questions
    survey_name = "Test Survey"
    survey = Survey(name=survey_name)
    for q in survey_questions:
        sq = SurveyQuestion()
        sq.question = Question(text=q)
        survey.questions.append(sq)
    session.add(survey)
    # set up model and a version
    model_description = 'Test Model'
    model = Model(description='Test Model')
    version = Version(description='Test Version')
    model.versions.append(version)
    session.add(model)
    # set up validation options
    validations_to_insert = []
    for v in range(2):
        validation = Validation(validation=int(v))
        validations_to_insert.append(validation)
    session.add_all(validations_to_insert)
    # set up prediction options
    predictions_to_insert = []
    for v in range(2):
        prediction = Validation(prediction=int(v))
        predictions_to_insert.append(prediction)
    session.add_all(predictions_to_insert)
    # set up respondents
    insert_respondents(df, respondent_attributes, session)
    # set up respones
    insert_responses(df, survey_questions, survey_name, model_description, session)
    session.commit()
