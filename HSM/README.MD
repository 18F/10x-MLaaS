# HSM
`app.py` provides a command line interface for the Ham-Spam Machine (HSM). Currently, it only fetches and classifies responses to the **site-wide version** of the survey. It uses a local instance of a postgreSQL database, although it does allow the user to validate comment predictions in excel while the script is sleeping.

## Getting Started

### Step 1: Clone the Repository
Navigate to where you'd like to clone the repo. Then clone it:
>`$ git clone https://github.com/18F/10x-MLaaS.git`

Now `cd` into the repository you've just cloned:
>`$ cd 10x-MLaaS`


### Step 2: Run with Docker & Docker Compose

#### Step a: Install Docker (if you do not have docker already)

https://docs.docker.com/compose/install/

#### Step b: Configure Environment Variables

>```
>$ cp sample.env .env
>```

Modify Qualtrics Env variables in `.env`:

>```
># In ~/10x-MLaaS/.env
>QUALTRICS_API_TOKEN=<token>
>QUALTRICS_SW_SURVEY_ID=<site_wide_survey_id>
>```

Install Pipenv if it is not installed:
```
pip install pipenv
```

Create a virtual environment with Pipenv to load environment variables locally to name the containers:
```
pipenv shell
```

If you ever made a new change to the `.env` file, you will need to restart your virtual environment and rebuild your containers:
Assuming you already started `pipenv shell` and you just updated your `.env`, then you can use the following commands:
```
exit # This will exit out of the virtual environmnet
pipenv shell
docker-compose build
```

#### Step 3 Build & Run!  

This will bring up the containers:
>```
>$ docker-compose up
>```

##### Using Qualtrics API
If the input exists in Qualtrics, you can run the command without any parameters.

(In another terminal).  This will run the application within `mlaas-web`:
>```
>$ docker exec --user hsm --workdir /home/hsm -it mlaas-web /bin/bash -c "python ~/HSM/app.py"
>```

##### Using Excel Spreadsheet
If the input is an Excel file (.xlsx), the Excel input file requires to be saved in `10x-MLaaS/HSM/model/inputs`.
You will need to only supply the filename in the command below. Replace <filename> with the actual filename.

(In another terminal).  This will run the application within `mlaas-web`:
>```
>$ docker exec --user hsm --workdir /home/hsm -it mlaas-web /bin/bash -c "python ~/HSM/app.py -i <filename>"
>```

i.e. The file path is `~/workspace/10x-MLaaS/HSM/model/inputs/survey.xlsx`, so the filename would be `survey.xlsx`.
The command would be as follows:
>```
>$ docker exec --user hsm --workdir /home/hsm -it mlaas-web /bin/bash -c "python ~/HSM/app.py -i survey.xlsx"
>```

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

 
## Load Data
There are times we need to load data in the database.  `load_data.py` will help with this task.  This version assumes it either has no `data` and `support_data` tables defined, or there are no data in `data` and `support_data` tables.  If they do exist, those will need to be deleted before performing running `load_data.py`, else, there can be duplicated data, incorrect representation of the actual data.

To use `load_data.py`, you can specify the path to the spreadsheet file that contains all the data.  The data will include two fields that represent the `filter_feature` and `validation`.
- `filter_feature` - This value is the filter feature that is used to run prediction on.  In this specific case, it is the `'Comments Concatenated'` field.
- `validation` - This value is the validation that started as a prediction and subject matter expert verified and corrected any mistake.  In this specific case, it is the `'Validation'`.

To run `load_data.py`, type in the following commands in the terminal (this assumes you have `docker_compose up` running):
```bash
docker exec --user hsm --workdir /home/hsm -it mlaas-web /bin/bash -c "cd HSM;python load_data.py <file>"
```


## TODO
 - log performance of the models (based on the ground truth established by the validation)
 - include a training data table in the database. Move training data there and include support to insert validated samples there.
 - retrain the classifier if a certain threshold of newly validated samples is met
 - extend to the page-level survey
 - include unit and integration tests
