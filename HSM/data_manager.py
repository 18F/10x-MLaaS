import requests
import zipfile
import json
import os
import sys
import pandas as pd
from time import sleep
from utils.config import qualtrics_sitewide_creds
from utils import db, db_utils


class DataManager():
    def __init__(self):
        pass

    def get_data(self):
        pass


class QualtricDataManager(DataManager):
    def __init__(self, last_response_id=None, apiToken=None, surveyId=None, fileFormat='json',
                 dataCenter='cemgsa'):
        # print("Getting data from Qualtrics...")
        print("Setting up Qualtric, and getting last response id")
        if not apiToken and not surveyId:
            apiToken = qualtrics_sitewide_creds['apiToken']
            surveyId = qualtrics_sitewide_creds['surveyId']

        self.apiToken = apiToken
        self.surveyId = surveyId
        self.fileFormat = fileFormat
        self.dataCenter = dataCenter
        if not last_response_id:
            db_utils.create_postgres_db()
            db.dal.connect()
            session = db.dal.Session()
            last_response_id = db_utils.fetch_last_RespondentID(session)
        self.lastResponseId = last_response_id

    def get_data(self):
        """
        Void function that gets and writes survey responses within the working
        directory.

        The process of getting survey responses requires four steps:
            1. Request the responses with the CreateResponseExport API.
            2. Request the export status with the GetResponseExportProgress API.
            3. Once the export progress is 100, make a GET request to retrieve
               the response file, which will be a zipped file.
            4. Unzip the file to find the survey responses in the format you
               requested (csv, csv2013, xml, json, or spss).

        Returns:
            None
        """

        # Setting static parameters
        print("Ready to get data...")
        requestCheckProgress = 0
        baseUrl = "https://{0}.gov1.qualtrics.com/API/v3/responseexports/".format(self.dataCenter)
        headers = {
                    "content-type": "application/json",
                    "x-api-token": self.apiToken,
                  }
        # Step 1: Creating Data Export
        downloadRequestUrl = baseUrl
        downloadRequestPayload = {
                                    "format": self.fileFormat,
                                    "surveyId": self.surveyId,
                                    "useLabels": True
                                 }
        # Include lastResponseId in payload if provided during init
        if self.lastResponseId:
            downloadRequestPayload['lastResponseId'] = self.lastResponseId

        # if start_date:
        downloadRequestPayload['startDate'] = "2018-08-01T14:47:02-05:00"  # 2018-08-01T13:47:05-05:00"#start_date

        # if end_date:
        downloadRequestPayload['endDate'] = "2019-06-27T12:40:20-05:00"  # end_date

        downloadRequestResponse = requests.request("POST", downloadRequestUrl,
                                                   data=json.dumps(downloadRequestPayload),
                                                   headers=headers)
        print('*'*80)
        print("Payload data")
        print(downloadRequestPayload)

        status_code = downloadRequestResponse.json()['meta']['httpStatus']
        if '200' in status_code:
            print('Post Request to Qualtrics was a success!')
        else:
            print(status_code)
            # TODO: log errors, including 500 status codes (see GH37)
            sys.exit(0)
        progressId = downloadRequestResponse.json()["result"]["id"]

        # Step 2: Checking on Data Export Progress and waiting until export is ready
        while requestCheckProgress < 100:
            sleep(2)
            requestCheckUrl = baseUrl + progressId
            print(requestCheckUrl)
            requestCheckResponse = requests.request("GET", requestCheckUrl, headers=headers)
            requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
            print("Download is " + str(requestCheckProgress) + " complete")

        # Step 3: Downloading file
        requestDownloadUrl = baseUrl + progressId + '/file'
        print(requestDownloadUrl)
        requestDownload = requests.request("GET", requestDownloadUrl,
                                           headers=headers, stream=True)

        # Step 4: Unzipping the file
        print("Upzipping file")
        with open("RequestFile.zip", "wb") as f:
            for chunk in requestDownload.iter_content(chunk_size=1024):
                f.write(chunk)
        zipfile.ZipFile("RequestFile.zip").extractall("temp")
        os.remove("RequestFile.zip")

        """
        Convert the json into a pandas dataframe
        """
        print("Converting json to pandas df")
        file_name = os.path.join(os.getcwd(), 'temp', 'USAgov Official Sitewide Survey.json')
        with open(file_name, encoding='utf8') as f:
            data = json.load(f)
        df = pd.DataFrame(data['responses'])
        # replace np.nan with None so sql insertions don't insert 'nan' strings
        df = df.where(pd.notnull(df), None)
        os.remove(file_name)
        df_n_rows = df.shape[0]
        # if number of rows more than zero
        if df_n_rows > 0:
            print("Ready to return")
            return df
        else:
            print("No new survey responses to download. Exiting")
            return None
