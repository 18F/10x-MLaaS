import json
import unittest
import base64
from api import app
from parameterized import parameterized
from unittest.mock import patch
from werkzeug.security import generate_password_hash


class BasicTestCase(unittest.TestCase):

    @parameterized.expand([
        ("index", "/"),
        ("predict", "/predict"),
        ("validate", "/validate"),
        ("train", "/train")
    ])
    def test_api_no_username(self, name, input):
        tester = app.test_client(self)
        response = tester.get(input)
        self.assertEqual(response.status_code, 401)

    @parameterized.expand([
        ("index", "/", {'title': 'Predict->Validate->Train', 'username': 'amy'}),
        ("predict", "/predict", {'task': 'predict', 'username': 'amy'}),
        ("validate", "/validate", {'task': 'validate', 'username': 'amy'}),
        ("train", "/train", {'task': 'train', 'username': 'amy'})
    ])
    @patch.dict('utils.config.users', {'amy': generate_password_hash('password')}, clear=True)
    def test_index(self, name, input, expected):
        tester = app.test_client(self)
        creds = base64.b64encode(b'amy:password').decode('utf-8')

        response = tester.get(input, headers={'Authorization': 'Basic ' + creds})
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(json.loads(response.data), expected)


if __name__ == '__main__':
    unittest.main()
