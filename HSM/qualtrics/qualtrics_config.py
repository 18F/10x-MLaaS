import os


# QUALTRICS API SETTINGS
apiToken = os.environ['QUALTRICS_API_TOKEN']
survey_id = os.environ['QUALTRICS_SW_SURVEY_ID']
qualtrics_sitewide_creds = {"apiToken":apiToken,
                            "surveyId":survey_id}