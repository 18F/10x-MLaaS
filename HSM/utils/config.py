import os

# DATABASE SETTINGS
DIALECT = "postgresql+psycopg2"
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']
DB_ADDR = os.environ['DB_ADDR']
DB_NAME = os.environ['DB_NAME']

SQLALCHEMY_URI = f"{DIALECT}://{DB_USER}:{DB_PASS}@{DB_ADDR}/{DB_NAME}"


# QUALTRICS API SETTINGS
apiToken = os.environ['QUALTRICS_API_TOKEN']
survey_id = os.environ['QUALTRICS_SW_SURVEY_ID']
qualtrics_sitewide_creds = {"apiToken": apiToken,
                            "surveyId": survey_id}
