import requests
import zipfile
import json
import io
import os
import sys
import pandas as pd
from time import sleep, strftime
import subprocess
import stat
import warnings
import json

warnings.filterwarnings('ignore')


class QualtricsApi:
    """Query Qualtrics API for new survey responses and then write to database.

    Attributes:
        apiToken (str): a Qualtrics API token.
        surveyId (str): the survey id.
        fileFormat (str): the preferred file format. Only 'csv' is possible now.
        dataCenter (str): the datacenter from the hostname of the qualtrics
                          account url
        SurveyResponsePath (str): the directory name to dump the responses
    """

    def __init__(self,
                 apiToken=None,
                 surveyId=None,
                 fileFormat='csv',
                 dataCenter='cemgsa',
                 SurveyResponsePath='survey_responses'):

        if not apiToken and not surveyId:
            with open('secrets.json','r') as f:
                loaded_json = json.loads(f.read())
                apiToken = loaded_json['apiToken']
                surveyId = loaded_json['surveyId']

        self.apiToken = apiToken
        self.surveyId = surveyId
        self.fileFormat = fileFormat
        self.dataCenter = dataCenter
        if not os.path.exists(SurveyResponsePath):
            os.makedirs(SurveyResponsePath)
        self.SurveyResponsePath = SurveyResponsePath
        try:
            with open('pastResponseId.txt','r') as f:
                line_list = f.read().splitlines()
                try:
                    self.lastResponseId = line_list[0]
                    self.penultimateResponseId = line_list[1]
                except IndexError:
                    self.penultimateResponseId = self.lastResponseId
        except FileNotFoundError:
            self.lastResponseId = None
            self.penultimateResponseId = None


    def download_responses(self):
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
        requestCheckProgress = 0
        baseUrl = "https://{0}.gov1.qualtrics.com/API/v3/responseexports/".format(self.dataCenter)
        headers = {"content-type": "application/json",
                   "x-api-token": self.apiToken,
                  }
        # Step 1: Creating Data Export
        downloadRequestUrl = baseUrl
        # Include lastResponseId in payload if provided during init
        if self.lastResponseId:
            downloadRequestPayload = '{"format":"' + self.fileFormat + \
                                     '","surveyId":"' + self.surveyId + \
                                     '","lastResponseId":"' + self.lastResponseId + '"}'
        else:
            downloadRequestPayload = '{"format":"' + self.fileFormat + \
                                     '","surveyId":"' + self.surveyId + '"}'

        downloadRequestResponse = requests.request("POST",downloadRequestUrl,
                                                   data=downloadRequestPayload,
                                                   headers=headers)

        status_code = downloadRequestResponse.json()['meta']['httpStatus']
        if '200' in status_code:
            print('Post Request to Qualtrics was a success!')
        else:
            print(status_code)
            sys.exit(0)
        progressId = downloadRequestResponse.json()["result"]["id"]

        # Step 2: Checking on Data Export Progress and waiting until export is ready
        while requestCheckProgress < 100:
            sleep(2)
            requestCheckUrl = baseUrl + progressId
            requestCheckResponse = requests.request("GET", requestCheckUrl, headers=headers)
            requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
            print("Download is " + str(requestCheckProgress) + " complete")

        # Step 3: Downloading file
        requestDownloadUrl = baseUrl + progressId + '/file'
        requestDownload = requests.request("GET", requestDownloadUrl,
                                            headers=headers, stream=True)

        # Step 4: Unzipping the file
        zipfile.ZipFile(io.BytesIO(requestDownload.content)).extractall(self.SurveyResponsePath)
        print('Complete!')

    def update_db(self):
        """
        Void function that walks through the response directory and concatenates
        (i.e. vertically stacks) all of the csv survey responses before writing
        to a single csv. Deletes the individual csvs after.
        """

        dfs = []
        for subdir, dirs, files in os.walk(self.SurveyResponsePath):
            for f in files:
                FilePath = os.path.join(subdir, f)

                csv_counter = 0
                if FilePath.endswith(".csv"):
                    #if it's the first csv, we'll need the column names
                    if csv_counter == 0:
                        df = pd.read_csv(FilePath)
                        # Get proper column names from first row
                        dfColumns = df.iloc[0].tolist()
                        df.columns = dfColumns
                        # Drop first two rows since they contain column info
                        df = df.iloc[2:]
                        dfs.append(df)
                        os.remove(FilePath)
                        csv_counter += 1
                    else:
                        df = pd.read_csv(FilePath)
                        df = df.iloc[2:]
                        dfs.append(df)
                        os.remove(FilePath)
                        csv_counter += 1
                else:
                    pass

        concat_df = pd.concat(dfs)
        # There are two ResponseID columns. Drop the second one
        cols = concat_df.columns
        second_id_i = [i for i,n in enumerate(cols) if n == 'ResponseID'][1]
        col_indices = [i for i,c in enumerate(cols) if i != second_id_i]
        final_df = concat_df.iloc[:, col_indices]

        # Rename some columns to something shorter
        col_rename_map = {k:None for k in final_df}
        for col in col_rename_map:
            if 'experience' in col:
                col_rename_map[col] = "Experience Rating"
            elif 'primary purpose' in col:
                col_rename_map[col] = "Purpose of Visit"
            elif 'few words' in col:
                col_rename_map[col] = "Other Purpose of Visit"
            elif 'able to accomplish' in col:
                col_rename_map[col]  = "Able to Accomplish"
            elif 'fully complete' in col:
                col_rename_map[col] = "Unable to Complete Purpose Reason"
            elif 'value most' in col:
                col_rename_map[col] = "Value"
            elif 'how we helped' in col:
                col_rename_map[col] = "Purpose"
            elif 'likely are you to return' in col:
                col_rename_map[col] = "Likely to Return"
            elif 'likely are you to recommend' in col:
                col_rename_map[col] = "Likely to Recommend"
            else:
                col_rename_map[col] = col
        final_df = final_df.rename(col_rename_map,axis=1)


        # Create new columns
        final_df['Download Date'] = pd.Timestamp.now()
        final_df['Value Spam'] = 1
        final_df['Purpose Spawm'] = 1
        final_df['Complete Spam'] = 1
        final_df['Other Spam'] = 1

        try:
            lastResponseId = final_df['ResponseID'].iat[-1]

            filename = 'pastResponseId.txt'
            if os.path.exists(filename):
                append_write = 'a' # append id if file already exists
                with open(filename,append_write) as f:
                    f.write(lastResponseId + "\n")
            else:
                append_write = 'w' # make a new file if it doesn't exist
                with open(filename,append_write) as f:
                    f.write(lastResponseId + "\n")
        except IndexError:
            print("No new survey responses to download!")
            sys.exit(0)

        # create or append to master db
        db_dir = os.path.join(os.getcwd(),f'db')
        db_path = os.path.join(db_dir,'db.csv')
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            final_df.to_csv(db_path, index=False)
        else:
            db = pd.read_csv(db_path)
            updated_db = pd.concat([db,final_df])
            updated_db = updated_db.drop_duplicates(subset='ResponseID')
            updated_db.to_csv(db_path, index=False)
