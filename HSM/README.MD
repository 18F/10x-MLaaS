# HSM
`app.py` provides a command line interface for the Ham-Spam Machine (HSM). Currently, it can fetches and classifies responses to the **site-wide version** and **page-level-version** of the surveys and possibly other surveys. It uses a local instance of a postgreSQL database to store the data for model retraining purpose. It also allows the user to validate comment predictions in Excel while the script is sleeping.

## Requirements
- [Docker](https://www.docker.com/)
- [Python](https://www.python.org/) (3.6)
- [pipenv](https://pipenv.kennethreitz.org/en/latest/)

## Limitations
- Currently we are in testing phase.  We have only tested to have about 18K rows of data to process.  If the data is larger than that, there is a chance that model cannot be trained as there is an issue that is unresolved at this point.  See [issue description here](https://github.com/18F/10x-MLaaS/issues/100).

## Getting Started

### Step 1: Clone the Repository
Navigate to where you'd like to clone the repo. Then clone it:
```bash
git clone https://github.com/18F/10x-MLaaS.git <Folder name if you do not want to use 10x-MLaaS, empty otherwise>
```

Now `cd` into the repository you've just cloned:
```bash
cd 10x-MLaaS # or the folder name you used above
```

### Step 2: Configure the application

#### Step a: Configure Environment Variables
An `.env` file is needed to specify settings on how to access Qualtrics API, how to set up Docker containers, and API.

First copy the template in `sample.env`.

```bash
cp sample.env .env
```

Modify environment variables to set up for the specific dataset in `.env`.  See details in `.env`.  If you want to store data on the cloud instead of using a local database and filesystem, see instructions on [Cloud.gov Configuration](cloud-gov-configuration).  This will allow you to have a database and filesystem on the cloud for sharing.

Install Pipenv if it is not installed:
```bash
pip install pipenv
```

Create a virtual environment with Pipenv to load environment variables locally to set up docker compose script to build and bring up your containers:
```bash
pipenv shell
```

If you ever made a new change to the `.env` file, you will need to restart your virtual environment and rebuild your containers.
Assuming you already started `pipenv shell` and you just updated your `.env`, then you can use the following commands:
```bash
exit # This will exit out of the virtual environment
pipenv shell
```

After that is done you will see `(10x-MLaaS)` or the name of the base folder if you have changed it in front of your command line prompt.

#### Step b: Configure Data Columns
There are specific settings that need to be set in `10x-MLaaS/HSM/utils/config.py` in order to know how to read the input dataset for prediction.  Those that has a label of `[ACTION]` will need to be updated according to your specific dataset.  These items are used to know how to create the classification results spreadsheet, which columns are used for filter feature, prediction, row identifier, and which columns are needed to process data.

### Step 3: Run with Docker & Docker Compose

#### Step a: Install Docker (if you do not have docker already)

https://docs.docker.com/compose/install/

#### Step b: Build & Run!

This will bring up the containers:
```bash
docker-compose up --build
```

##### Input dataset via Qualtrics API
If the input exists in Qualtrics, you can run the command without any parameters.

(In another terminal).  This will run the application within `${CONTAINER_NAME_WEB}`.  `${CONTAINER_NAME_WEB}` is an
environment variable you have specified in your `.env` file.
```bash
pipenv shell
docker exec --user hsm --workdir /home/hsm -it ${CONTAINER_NAME_WEB} /bin/bash -c "python ~/HSM/app.py"
```

##### Input dataset is an Excel Spreadsheet
If the input is an Excel file (.xlsx), the Excel input file requires to be saved in `10x-MLaaS/HSM/model/inputs`.
You will need to only supply the filename in the command below. Replace <filename> with the actual filename.

(In another terminal).  This will run the application within `${CONTAINER_NAME_WEB}`  `${CONTAINER_NAME_WEB}` is an
environment variable you have specified in your `.env` file:
```bash
pipenv shell
docker exec --user hsm --workdir /home/hsm -it ${CONTAINER_NAME_WEB} /bin/bash -c "python ~/HSM/app.py -i <filename>"
```

i.e. The file path is `~/workspace/10x-MLaaS/HSM/model/inputs/survey.xlsx`, so the filename would be `survey.xlsx`.
The command would be as follows:
```bash
docker exec --user hsm --workdir /home/hsm -it ${CONTAINER_NAME_WEB} /bin/bash -c "python ~/HSM/app.py -i survey.xlsx"
```

## Using the CLI
Now that your environment is set up, you can use `app.py` as a CLI tool. Here's what that script does:
 - Downloads data from the Qualtrics API. 
    - If it's your first time running this, it'll download all responses to-date. Otherwise it'll check the database for the last response and then only fetch new responses.
 - Feeds concatenated survey comments to a pre-trained `sklearn` classifer to predict spam (1) or ham (0)
 - User can make changes to what fields to return as part of the outputting spreadsheet in utils/config.py
   - Keep in mind that you should make sure `SPAM` and `Comment Concatentated` fields should be included for training purpose later.
 - Sleeps to give you time to review the predictions in  `HSM/model/results/ClassificationResults.xlsx`
    - When reviewing the results, the prediction is in the `SPAM` column (0 = ham and 1 = spam). 
    - Make your changes inplace, overwriting the prediction if you disagree.
    - Save and exit the file once you're done. Do not alter the file name.
    - Return to your terminal and enter `y` to tell the script to wake up and continue.

 - Inserts the survey data along with model predictions and your validation into the database.


## Cloud.gov Configuration
These are work-in-progress instructions as they were only quickly tested out.  You will need to swap out `docker-compose.yml` to use the `docker-compose-cloud.yml` file first.  Save the current `docker-compose.yml` at a safe place that you can swap back out if necessary when you want to use database within Docker.  The following commands include moving `docker-compose.yml` to `docker-compose-local.yml`:
```bash
mv docker-compose.yml docker-compose-local.yml
mv docker-compose-cloud.yml docker-compose.yml
```

### Cloud.gov Database
This tool makes use of a database to store all the dataset for the purposes of training the model, keeping all the available data to avoid downloading the same data from Qualtrics again.  In order to use a database on Cloud.gov, you will need to have an account on cloud.gov, and you will create an application with a Postgresql database service instance.  This service instance is also binded to the application you create.  Once that is set up, you will need to log into cloud.gov in your terminal:
```bash
cf login -a api.fr.cloud.gov --sso
```

You will need to target your organization and space where your application is in.  Then you will need to install [CF-Service-Connect](https://github.com/18F/cf-service-connect) if you haven't.  This will allow you to SSH tunnel into Cloud.gov environment so you will have access to your database.  Once installed, you will need to connect to the database by running the following command, and replacing those in the `< >`:
```bash
cf connect-to-service -no-client <cloud.gov app> <cloud.gov database binded to the app>
```

You will get an output that looks like this:
```bash
It will give you the following output, pull the corresponding information out as environment variables:
Finding the service instance details...
Setting up SSH tunnel...
SSH tunnel created.
Skipping call to client CLI. Connection information:

Host: localhost
Port: <CLOUD_DB_PORT>
Username: <CLOUD_DB_USER>
Password: <CLOUD_DB_PASS>
Name: <CLOUD_DB_NAME>

Leave this terminal open while you want to use the SSH tunnel. Press Control-C to stop.
```

Take the information in the `< >` above and modify the `.env` file accordingly.  You will have to keep this terminal window open throughout the time you run the tool.

If at any time, you are getting errors about the database access, you may have to exit out of this and rerun the `connect-to-service` command.

### Cloud.gov S3 Storage
This part has not been tested out at all.  The storage is to store the machine learning model.  But you will do the same as you would set up the Cloud.gov database.  You will need to create a new S3 service instance, and bind it to an application.  This is the [detailed instructions on how to interact with S3 outside of Cloud.gov](https://cloud.gov/docs/services/s3/#interacting-with-your-s3-bucket-from-outside-cloud-gov). Code changes are necessary to get this implemented.

 
## Load Data
There are times we need to load data in the database.  `load_data.py` will help with this task.  This version assumes it either has no `data` and `support_data` tables defined, or there are no data in `data` and `support_data` tables.  If they do exist, those will need to be deleted before performing running `load_data.py`, else, there can be duplicated data, incorrect representation of the actual data.

To use `load_data.py`, you can specify the path to the spreadsheet file that contains all the data.  The data will include two fields that represent the `filter_feature` and `validation`.
- `filter_feature` - This value is the filter feature that is used to run prediction on.  In this specific case, it is the `'Comments Concatenated'` field.
- `validation` - This value is the validation that started as a prediction and subject matter expert verified and corrected any mistake.  In this specific case, it is the `'Validation'`.

To run `load_data.py`, type in the following commands in the terminal (this assumes you have `docker_compose up` running and your command line prompt activated
the virtual environment with `pipenv shell`):
```bash
docker exec --user hsm --workdir /home/hsm -it ${CONTAINER_NAME_WEB} /bin/bash -c "cd HSM;python load_data.py <file>"
```

## Testing
The current state of the code do not included a lot of automated testing.  During development, to ensure the tool is still working as intended, a manual testing process is necessary to spot check different items.  If you are the general user and not the person who maintain the tool.  You will need to work with the maintainer if any issues come up.

### End-to-end testing
Assuming that you have already set up the tool to run prediction/training.  These are the steps to follow after to do end-to-end testing:

#### Pulling the data
- You would need to follow [Step 3](#step-3-run-with-docker--docker-compose) under [Getting Started](#getting-started) to build and run the tool
- Once the tool pulled the data, you will make sure it didn't fail prematurely.

#### Prediction
- Once it performed prediction, you will be prompted to check the prediction on the spreadsheet
- Open the spreadsheet, you would want to check the field that you have specified in `config.py` as `PREDICTION_FIELD_NAME`.  To check if data looks okay, and correct them accordingly.
- Make sure all the data columns you specified in `config.py` as `FIELDS` are included.
- There would be columns named as what you specified in `config.py` as `FILTER_FEATURE`, `NORMALIZED_FILTER_FEATURE`, `PREDICTION_FIELD_NAME`, and `ENTRY_ID`.
- The column with the name specified in `config.py` as `FILTER_FEATURE` should have essential the combination of all of columns in `FILTER_FEATURE_FIELDS`.  They are not directly the same values because the `FILTER_FEATURE` has been processed to pull out some unnecessary words for training purposes.  But do a few spot check from the data to see it captures most of the essential words.

#### Training
- Once you type `Y` to say the prediction is correct, it will start inserting data into the database, note any errors that come out.
- Once it inserted all the data, it will start the retraining process.  It should run smoothly with a `Classification Report` that comes out, an ended with a `DONE!`. If at any point, it started hanging, there may be an issue.


## TODO
 - log performance of the models (based on the ground truth established by the validation)
 - include a training data table in the database. Move training data there and include support to insert validated samples there.
 - retrain the classifier if a certain threshold of newly validated samples is met
 - extend to the page-level survey
 - include unit and integration tests
