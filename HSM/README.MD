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


#### Step 3 Build & Run!  

This will bring up the containers:
>```
>$ docker-compose up
>```
  
(In another terminal).  This will run the application within `mlaas-web`:
>```
>$ docker exec --user hsm --workdir /home/hsm -it mlaas-web /bin/bash -c "python ~/HSM/app.py"
>```


## Using the CLI
Now that you're environment is set up, you can use `app.py` as a CLI tool. Here's what that script does:
 - Downloads data from the Qualtrics API. 
    - If it's your first time running this, it'll download all responses to-date. Otherwise it'll check the database for the last response and then only fetch new responses.
 - Feeds concatenated survey comments to a pre-trained `sklearn` classifer to predict spam (1) or ham (0)
 - Sleeps to give you time to review the predictions in  `HSM/model/results/ClassificationResults.xlsx`
    - When reviewing the results, the prediction is in the `SPAM` column (0 = ham and 1 = spam). 
    - Make your changes inplace, overwriting the prediction if you disagree.
    - Save and exit the file once you're done. Do not alter the file name.
    - Return to your terminal and enter `y` to tell the script to wake up and continue.

 - Inserts the survey data along with model predictions and your validation into the database.

 
## TODO
 - log performance of the models (based on the ground truth established by the validation)
 - include a training data table in the database. Move training data there and include support to insert validated samples there.
 - retrain the classifier if a certain threshold of newly validated samples is met
 - extend to the page-level survey
 - include unit and integration tests
