import os
from werkzeug.security import generate_password_hash

# FLASK SETTINGS
APP_SECRET_KEY = os.environ['APP_SECRET_KEY']
APP_PORT = os.getenv("PORT", 8080)

# USER SETTINGS
users = {
    "admin": generate_password_hash(os.getenv("ADMIN")),
    "user": generate_password_hash(os.getenv("USER"))
}

# DATABASE SETTINGS
print(f'CLOUD_GOV:{os.getenv("CLOUD_GOV")}')
DB_DIALECT = "postgresql+psycopg2"
if not os.getenv('CLOUD_GOV') or os.getenv('CLOUD_GOV') == 'NO':
    DB_USER = os.getenv('DB_USER')
    DB_PASS = os.getenv('DB_PASS')
    DB_ADDR = os.getenv('DB_ADDR')
    DB_NAME = os.getenv('DB_NAME')

    SQLALCHEMY_URI = f"{DB_DIALECT}://{DB_USER}:{DB_PASS}@{DB_ADDR}/{DB_NAME}"
else:  # CLOUD_GOV
    CLOUD_DB_USER = os.getenv('CLOUD_DB_USER')
    CLOUD_DB_PASS = os.getenv('CLOUD_DB_PASS')
    CLOUD_DB_ADDR = os.getenv('CLOUD_DB_ADDR')
    CLOUD_DB_PORT = os.getenv('CLOUD_DB_PORT')
    CLOUD_DB_NAME = os.getenv('CLOUD_DB_NAME')
    SQLALCHEMY_URI = f"{DB_DIALECT}://{CLOUD_DB_USER}:{CLOUD_DB_PASS}@{CLOUD_DB_ADDR}:{CLOUD_DB_PORT}/{CLOUD_DB_NAME}"
    print(f'SQLALCHEMY_URI: {SQLALCHEMY_URI}')


# QUALTRICS API SETTINGS
apiToken = os.environ['QUALTRICS_API_TOKEN']
survey_id = os.environ['QUALTRICS_SW_SURVEY_ID']
filename = os.environ['QUALTRICS_FILENAME']
if filename:
    filename = filename.replace('"', '').replace("'", "")

qualtrics_sitewide_creds = {
                            "apiToken": apiToken,
                            "surveyId": survey_id,
                            "filename": filename,
                            }

# INPUT FILE SETTINGS
INPUT_DIR = os.path.join(os.getcwd(), 'HSM', 'model', 'inputs')
if not os.path.exists(INPUT_DIR):
    os.makedirs(os.path.join(INPUT_DIR))

# CELERY SETTINGS

# SPREADSHEET SETTINGS

# [ACTIONS] These are fields that need to be updated for your input dataset.
# The list of fieldnames that you want to appear in your ClassificationResults.xlsx
FIELDS = [
    "ResponseID",
    "pageType",
    "StartDate",
    "EndDate",
    "Country",
    "State",
    "UPVC",
    "TVPC",
    "Site_Referrer",
    "PR_URL",
    "CP_URL",
    "Asset_Click",
    "Q1",
    "Q4",
    "Comments Concatenated",
    "SPAM",
    "Q2",
    "Q3",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "DeviceType",
    "Referer",
    "History",
    "Browser Metadata Q_1_TEXT",
    "Browser Metadata Q_2_TEXT",
    "Browser Metadata Q_3_TEXT",
    "Browser Metadata Q_4_TEXT",
    # "Duration (in seconds)",
]

# DATA COLUMNS SETTINGS

# [ACTIONS] Fields are needed to do filter, they will be combined to be used for prediction and training
FILTER_FEATURE_FIELDS = ['Q5', 'Q7', 'Q6', 'Q3']

# [ACTIONS] Fielded from the raw spreadsheet to be included when processing data,
# key is the field name we want to use for the processed data dataframe
# value is the field name being used in the raw data
FIELDS_TO_INCLUDED_FOR_PROCESSED_DATA_MAPPING = {'ResponseID': 'ResponseID', 'Date': 'EndDate'}

# [ACTIONS] Field name that represents the filter feature
FILTER_FEATURE = 'Comments Concatenated'

# [ACTIONS] Field name that represents the normalized filter feature
NORMALIZED_FILTER_FEATURE = 'Normalized Comments'

# [ACTIONS] Field name to place the prediction/validation
PREDICTION_FIELD_NAME = 'SPAM'

# [ACTIONS] Field name that identify each row in the raw spreadsheet
ENTRY_ID = 'ResponseID'
