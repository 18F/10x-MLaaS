import json
import pandas as pd
from utils.config import SQLALCHEMY_URI
from sqlalchemy_utils import database_exists, create_database
from utils.db import (
    Data,
    SupportData,
)
from utils.config import ENTRY_ID
from sqlalchemy import create_engine  # , desc, event, func
from sqlalchemy import func


def create_postgres_db():
    connection_string = SQLALCHEMY_URI
    engine = create_engine(connection_string, echo=False)
    if not database_exists(engine.url):
        create_database(engine.url)


def insert_data(df, session):
    '''
    Insert data and supporting data into database
    '''

    filter_feature = 'Comments_Concatenated'
    validation = 'validated prediction'
    data_columns = [filter_feature, validation]

    data = df[data_columns]
    support_data = json.loads(df[df.columns.difference(data_columns)].to_json(orient='records'))

    for i in range(len(data)):

        data_row = data.iloc[i]
        support_data_row = support_data[i]

        data_obj = Data(filter_feature=str(data_row[filter_feature]), validation=int(data_row[validation]))
        session.add(data_obj)
        session.flush()
        support_data_obj = SupportData(support_data=support_data_row)
        data_obj.support_data = support_data_obj
        support_data_obj.data = data_obj
        support_data_obj.data_id = support_data_obj.data.id
        session.add(support_data_obj)

    session.commit()


def get_data(session, filter_feature='filter_feature', validation='validation'):
    '''
    Get data from database and return as dataframe
    '''

    data_rows = [(row.filter_feature, row.validation) for row in session.query(Data).all()]
    df = pd.DataFrame(data_rows, columns=[filter_feature, validation])
    return df


def fetch_last_RespondentID(session):
    '''
    Fetch the last RespondentID from the database to use with the Qualtrics API

    Parameters:
        session: an instance of a sqlalchemy session object created by DataAccessLayer

    Returns:
        last_response_id (str): the RespondentID of the last survey response
    '''
    try:
        last_response = session.query(Data).order_by(Data.id.desc()).first()
        last_response_id = last_response.support_data.support_data[ENTRY_ID]

    except AttributeError:
        last_response_id = None

    return last_response_id


def count_table_rows(table, session):
    rows = session.query(func.count(table.id)).scalar()

    return rows
