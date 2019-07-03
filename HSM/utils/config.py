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
