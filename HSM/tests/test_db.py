import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.db_utils import fetch_concatenated_comments, prep_test_db
from db.db_config import SQLALCHEMY_URI


# global application scope.  create Session class, engine
Session = sessionmaker()

engine = create_engine(SQLALCHEMY_URI)

class SomeTest(unittest.TestCase):
    def setUp(self):
        # connect to the database
        self.connection = engine.connect()

        # begin a non-ORM transaction
        self.trans = self.connection.begin()

        # bind an individual Session to the connection
        self.session = Session(bind=self.connection)

    def test_something(self):
        # use the session in tests.

        self.session.add(Foo())
        self.session.commit()

    def tearDown(self):
        self.session.close()

        # rollback - everything that happened with the
        # Session above (including calls to commit())
        # is rolled back.
        self.trans.rollback()

        # return connection to the Engine
        self.connection.close()