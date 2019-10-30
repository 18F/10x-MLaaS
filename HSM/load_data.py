import json
from argparse import ArgumentParser
import pandas as pd
from utils import db, db_utils
from utils.db import Data, SupportData

filter_feature = 'Comments Concatenated'
validation = 'Validation'


def main(file):
    db_utils.create_postgres_db()
    db.dal.connect()
    session = db.dal.Session()

    df = pd.read_excel(file)

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
        print(f'data: {data_obj.id}')
        print(f'support_data: {support_data_obj.id}')
        print(f'support_data.data: {support_data_obj.data.id}')

    session.commit()


if __name__ == '__main__':

    program_desc = '''This application will get the spreadsheet and pull out essential data to fill out
                      the database. It will populate the database in the `data` table.  It also put all
                      other data in the database as well in support_data table.'''

    parser = ArgumentParser(description=program_desc)
    parser.add_argument("file", help="specify path to file")

    args = parser.parse_args()

    main(file=args.file)
