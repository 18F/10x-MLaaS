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
if not os.getenv('CLOUD_GOV'):
    DIALECT = "postgresql+psycopg2"
    DB_USER = os.getenv('DB_USER')
    DB_PASS = os.getenv('DB_PASS')
    DB_ADDR = os.getenv('DB_ADDR')
    DB_NAME = os.getenv('DB_NAME')

    SQLALCHEMY_URI = f"{DIALECT}://{DB_USER}:{DB_PASS}@{DB_ADDR}/{DB_NAME}"
else:  # CLOUD_GOV
    SQLALCHEMY_URI = os.getenv('DATABASE_URL')
    if "postgresql:" in SQLALCHEMY_URI:  # This means the dialect is not included
        SQLALCHEMY_URI = SQLALCHEMY_URI.replace("postgresql:" "postgresql+psycopg2:", 1)


# QUALTRICS API SETTINGS
apiToken = os.environ['QUALTRICS_API_TOKEN']
survey_id = os.environ['QUALTRICS_SW_SURVEY_ID']
qualtrics_sitewide_creds = {"apiToken": apiToken,
                            "surveyId": survey_id}

# CELERY SETTINGS

# SPREADSHEET SETTINGS
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
FILTER_FEATURE = 'Comments Concatenated'
VALIDATION = 'Validation'
ENTRY_ID = 'ResponseID'
