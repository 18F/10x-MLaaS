from sqlalchemy import create_engine, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, Table, Text, \
                       Date, Boolean, Sequence, DateTime
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import desc
from config import SQLALCHEMY_URI
from sqlalchemy import func
import pandas as pd

Base = declarative_base()

class DataAccessLayer:

    def __init__(self):
        self.engine = None
        self.conn_string = SQLALCHEMY_URI

    def connect(self):
        self.engine = create_engine(self.conn_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

dal = DataAccessLayer()

class Survey(Base):
    __tablename__ = 'survey'
    
    id = Column(Integer, primary_key = True)
    name = Column(String(100))
    
    questions = relationship("SurveyQuestion", back_populates="survey")
    

class Question(Base):
    __tablename__ = 'question'
    
    id = Column(Integer, primary_key=True)
    text = Column(String(1000))
    
    surveys = relationship("SurveyQuestion", back_populates="question")


class SurveyQuestion(Base):
    '''
    Bi-directional association table for many-to-many relationship b/w Survey and Question.
    '''
    __tablename__ = 'survey_question'
    
    survey_id = Column(Integer, ForeignKey('survey.id'), primary_key = True)
    question_id = Column(Integer, ForeignKey('question.id'), primary_key = True)
    
    survey = relationship("Survey", back_populates="questions")
    question = relationship("Question", back_populates="surveys")
    

class Respondent(Base):
    __tablename__ = 'respondent'

    id = Column(Integer, primary_key=True)
    Browser_Metadata_Q_1_TEXT = Column(String(1000))
    Browser_Metadata_Q_2_TEXT = Column(String(1000))
    Browser_Metadata_Q_3_TEXT = Column(String(1000))
    Browser_Metadata_Q_4_TEXT = Column(String(1000))
    Browser_Metadata_Q_5_TEXT = Column(String(1000))
    Browser_Metadata_Q_6_TEXT = Column(String(1000))
    Browser_Metadata_Q_7_TEXT = Column(String(1000))
    CP_URL = Column(String(10000))
    Country = Column(String(1000))
    DeviceType = Column(String(1000))
    EndDate = Column(String(1000))
    ExternalDataReference = Column(String(1000))
    Finished = Column(String(1000))
    History = Column(String(10000))
    IPAddress = Column(String(1000))
    LocationAccuracy = Column(String(1000))
    LocationLatitude = Column(String(1000))
    LocationLongitude = Column(String(1000))
    PR_URL = Column(String(10000))
    RecipientEmail = Column(String(1000))
    RecipientFirstName = Column(String(1000))
    RecipientLastName = Column(String(1000))
    Referer = Column(String(10000))
    ResponseID = Column(String(1000))
    ResponseSet = Column(String(1000))
    SPAM = Column(String(1000))
    Site_Referrer = Column(String(10000))
    StartDate = Column(String(1000))
    State = Column(String(1000))
    Status = Column(String(1000))
    TVPC = Column(String(1000))
    UPVC = Column(String(1000))
    UserAgent = Column(String(1000))
    Welcome_Text = Column(String(1000))
    pageType = Column(String(1000))
    
    responses = relationship("Response")


class Response(Base):
    __tablename__ = 'response'
    
    id = Column(Integer, primary_key = True)
    respondent_id = Column(Integer, ForeignKey('respondent.id'))
    survey_id = Column(Integer, ForeignKey('survey.id'))
    question_id = Column(Integer, ForeignKey('question.id'))
    validation_id = Column(Integer, ForeignKey('validation.id'))
    text = Column(String(10000), nullable = True)
    
    version_predictions = relationship("VersionPrediction")  

    
class Model(Base):
    
    __tablename__ = 'model'
    
    id = Column(Integer, primary_key = True)
    description = Column(String(100))
    
    versions = relationship("Version")

    
class Version(Base):
    
    __tablename__ = "version"
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('model.id'))
    description = Column(String(100))
    
    predictions = relationship("VersionPrediction", back_populates="version")
    

class Prediction(Base):
    
    __tablename__ = 'prediction'
    
    id = Column(Integer, primary_key = True)
    prediction = Column(Integer)
    
    versions = relationship("VersionPrediction", back_populates="prediction")
    

class VersionPrediction(Base):
    '''
    Bi-directional association table for many-to-many relationship b/w Version and Prediction.
    '''
    __tablename__ = 'version_prediction'

    model_id = Column(Integer, ForeignKey('model.id', primary_key = True))
    version_id = Column(Integer, ForeignKey('version.id'), primary_key = True)
    prediction_id = Column(Integer, ForeignKey('prediction.id'), primary_key = True)
    response_id = Column(Integer, ForeignKey('response.id'), primary_key = True)

    version = relationship("Version", back_populates="predictions")
    prediction = relationship("Prediction", back_populates="versions")


class Validation(Base):
    
    __tablename__ = 'validation'
    
    id = Column(Integer, primary_key = True)
    validation = Column(Integer)
    
    responses = relationship("Response")
    

def create_postgres_db():
    connection_string = SQLALCHEMY_URI
    engine = create_engine(connection_string, echo=False)
    if not database_exists(engine.url):
        print("Database doesn't exist, creating now...")
        create_database(engine.url)
        print("Done creating postgresql database.")
    else:
        print("Database already exists!")
    

def survey_exists(survey_name, survey_questions, session):
    survey_names = []
    for row in session.query(Survey.name).all():
        survey_names.append(row.name)

    if survey_name in survey_names:
        pass
    else:
        site_wide = Survey(name = "Site-Wide Survey English")
        for q in survey_questions:
            sq = SurveyQuestion()
            sq.question = Question(text=q)
            site_wide.questions.append(sq)
        session.add(site_wide)
        
def model_exists(model_description, session):
    model_description = model_description.replace(".pkl","")
    version_description = model_description.rsplit('_', 1)[-1]
    model_description = model_description.replace("_"+version_description,"")
    version = Version(description = version_description)
    model_descriptions = []
    for row in session.query(Model.description).all():
        model_descriptions.append(row.description)
    if model_description in model_descriptions:
        pass
    else:
        model = Model(description = model_description)
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
    for i in range(df.shape[0]):
        data = df.iloc[i][respondent_attributes]
        respondent_data = {k:v for k,v in zip(respondent_attributes, data)}
        respondent = Respondent(**respondent_data)
        session.add(respondent)      
    

def insert_responses(df, survey_questions, survey_name, model_description, session):
    
    def fetch_survey_id(survey_name, session):
        survey_id = session.query(Survey.id).filter(Survey.name==survey_name).first().id
        return survey_id

    def fetch_question_id(question_name, session):
        question_id = session.query(Question.id).filter(Question.text==question_name).first().id
        return question_id

    def fetch_respondent_id(qualtrics_response_id, session):
        respondent_id = session.query(Respondent.id).filter(Respondent.ResponseID==qualtrics_response_id).first().id
        return respondent_id 

    def fetch_validation_id(validation, session):
        validation_id = session.query(Validation.id).filter(Validation.validation==validation).first().id
        return validation_id

    def fetch_model_version_ids(model_description, session):
        model_description = model_description.replace(".pkl","")
        version_description = model_description.rsplit('_', 1)[-1]
        model_description = model_description.replace("_"+version_description,"")
        model_id = session.query(Model.id).filter(Model.description==model_description).first().id
        version_id = session.query(Version.id).filter(Version.description==version_description).first().id
        
        return model_id, version_id
    
    def fetch_prediction_id(prediction, session):
        prediction_id = session.query(Prediction.id).filter(Prediction.prediction==prediction).first().id
        
        return prediction_id

    survey_id = fetch_survey_id(survey_name, session)
    model_id, version_id = fetch_model_version_ids(model_description, session)
    
    for i in range(df.shape[0]):
        data = df.iloc[i][survey_questions+['ResponseID','prediction','validated prediction']]
        respondent_id = fetch_respondent_id(data['ResponseID'], session)
        validation = int(data['validated prediction'])
        validation_id = fetch_validation_id(validation, session)
        pred = int(data['prediction'])
        prediction_id = fetch_prediction_id(pred, session)
        prediction = session.query(Prediction).get(prediction_id)
        val = session.query(Validation).get(validation_id)
        for q in survey_questions:
            question_id = fetch_question_id(q, session)
            response = Response(survey_id = survey_id,
                                question_id = question_id,
                                respondent_id = respondent_id, 
                                text=data[q])
            val.responses.append(response)
            session.add(response)
            session.flush()
            response_id = response.id
            version_prediction = VersionPrediction()
            version_prediction.model_id = model_id
            version_prediction.version_id = version_id
            version_prediction.prediction_id = prediction_id
            version_prediction.response_id = response_id
            version_prediction.prediction = prediction
            session.add(version_prediction)   


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
        question_id = session.query(Question.id).filter(Question.text==question_name).first().id
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
    respondent_attributes = ['Browser_Metadata_Q_1_TEXT','Browser_Metadata_Q_2_TEXT',
                             'Browser_Metadata_Q_3_TEXT','Browser_Metadata_Q_4_TEXT',
                             'Browser_Metadata_Q_5_TEXT','Browser_Metadata_Q_6_TEXT',
                             'Browser_Metadata_Q_7_TEXT','CP_URL','Country','DeviceType',
                             'EndDate','ExternalDataReference','Finished','History','IPAddress',
                             'LocationAccuracy','LocationLatitude','LocationLongitude','PR_URL',
                             'RecipientEmail','RecipientFirstName','RecipientLastName','Referer',
                             'ResponseID','ResponseSet','SPAM','Site_Referrer','StartDate','State',
                             'Status','TVPC','UPVC','UserAgent','Welcome_Text','pageType',
                             'Comments_Concatenated']
    pred_cols = ['prediction', 'validated prediction']
    cols = survey_questions+respondent_attributes+pred_cols
    d = {k:['abc', '123'] for k in cols}
    df = pd.DataFrame(d)
    #set up survey and questions
    survey_name = "Test Survey"
    survey = Survey(name = survey_name)
    for q in survey_questions:
        sq = SurveyQuestion()
        sq.question = Question(text=q)
        survey.questions.append(sq)
    session.add(survey)
    #set up model and a version
    model_description = 'Test Model'
    model = Model(description = 'Test Model')
    version = Version(description = 'Test Version')
    model.versions.append(version)
    session.add(model)
    #set up validation options
    validations_to_insert = []
    for v in range(2):
        validation = Validation(validation=int(v))
        validations_to_insert.append(validation)
    session.add_all(validations_to_insert)
    #set up prediction options
    predictions_to_insert = []
    for v in range(2):
        prediction = Palidation(prediction=int(v))
        predictions_to_insert.append(prediction)
    session.add_all(predictions_to_insert)
    #set up respondents
    insert_respondents(df, respondent_attributes, session)
    #set up respones
    insert_responses(df, survey_questions, survey_name, model_description, session)
    session.commit()

  