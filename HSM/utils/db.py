from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from utils.config import SQLALCHEMY_URI

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

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))

    questions = relationship("SurveyQuestion", back_populates="survey")


class Question(Base):
    __tablename__ = 'question'

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(1000))

    surveys = relationship("SurveyQuestion", back_populates="question")


class SurveyQuestion(Base):
    '''
    Bi-directional association table for many-to-many relationship b/w Survey and Question.
    '''
    __tablename__ = 'survey_question'

    survey_id = Column(Integer, ForeignKey('survey.id'), primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey('question.id'), primary_key=True, index=True)

    survey = relationship("Survey", back_populates="questions")
    question = relationship("Question", back_populates="surveys")


class Respondent(Base):
    __tablename__ = 'respondent'

    id = Column(Integer, primary_key=True, index=True)
    Asset_Click = Column(String(1000))
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

    id = Column(Integer, primary_key=True, index=True)
    respondent_id = Column(Integer, ForeignKey('respondent.id'))
    survey_id = Column(Integer, ForeignKey('survey.id'))
    question_id = Column(Integer, ForeignKey('question.id'))
    validation_id = Column(Integer, ForeignKey('validation.id'))
    text = Column(String(10000), nullable=True)

    version_predictions = relationship("VersionPrediction")


class Model(Base):

    __tablename__ = 'model'

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String(100))

    versions = relationship("Version")


class Version(Base):

    __tablename__ = "version"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey('model.id'))
    description = Column(String(100))

    predictions = relationship("VersionPrediction", back_populates="version")


class Prediction(Base):

    __tablename__ = 'prediction'

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(Integer)

    versions = relationship("VersionPrediction", back_populates="prediction")


class VersionPrediction(Base):
    '''
    Bi-directional association table for many-to-many relationship b/w Version and Prediction.
    '''
    __tablename__ = 'version_prediction'

    model_id = Column(Integer, ForeignKey('model.id', primary_key=True), index=True)
    version_id = Column(Integer, ForeignKey('version.id'), primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey('prediction.id'), primary_key=True, index=True)
    response_id = Column(Integer, ForeignKey('response.id'), primary_key=True, index=True)

    version = relationship("Version", back_populates="predictions")
    prediction = relationship("Prediction", back_populates="versions")


class Validation(Base):

    __tablename__ = 'validation'

    id = Column(Integer, primary_key=True, index=True)
    validation = Column(Integer)

    responses = relationship("Response")
