import unittest
from db import fetch_concatenated_comments, prep_test_db, dal


class TestApp(unittest.TestCase):
    

    @classmethod
    def setUpClass(cls):
        #ToDo: import from config # dal.conn_string = 
        dal.connect()
        dal.session = dal.Session()
        prep_test_db(dal.session)
        dal.session.close()

    def setUp(self):
        dal.session = dal.Session()

    def tearDown(self):
        dal.session.rollback()
        dal.session.close()

    def test_fetch_concatenated_comments(self):
        results = fetch_concatenated_comments()
        self.assertEqual(results, [])