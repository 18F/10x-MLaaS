# HSM
`app.py` provides a command line interface for the Ham-Spam Machine (HSM). Currently, it only fethces and classifies responses to the **site-wide version** of the survey. It uses a local instance of a postgreSQL database, although it does allow the user to validate comment predictions in excel while the script is sleeping. 

## Getting Started

### Step 1: Install & Configure PostgresSQL

#### Step 1.1:  Install Homebrew
You don't have to install homebrew in order to install postgres. But it's by far the easiest approach.

>`$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

#### Step 1.2: Check Homebrew
Before you install anything with Homebrew, you should always make sure it's up to date and that it's healthy:
>`$ brew update`<br>
>`$ brew doctor`

#### Step 1.3: Brew Install Postgres
>`$ brew install postgresql`

When you install Postgres, you will see a bunch of output in your Terminal.

#### Step 1.4: Check Your Installation
>`$ pg_config`

This will print a lot of information to your terminal if postgres has been successfully installed.


### Step 2: Clone the Repository
Navigate to where you'd like to clone the repo. Then clone it:
>`$ git clone https://github.com/18F/10x-qualitative-data.git`

Now `cd` into the repository you've just cloned:
>`$ cd 10x-qualitative-data`

### Step 3: Install Python Requirements
This project uses Python 3.6.5 and `venv` to manage Python dependencies. Most Python 3.x versions should be fine.

#### Step 3.1: Create a Virtual Environment:

>`$ python3 -m venv env`

#### Step 3.2: Activate the Environment:

>`$ source env/bin/activate`

Notice how your prompt is now prefixed with the name of your environment (env, in our case). This is the indicator that env is currently active, which means the python executable will only use this environment’s packages and settings. You can deactivate the environment with `$ deactivate`.

#### Step 3.3: Install Requirements

>`$ pip install -r requirements.txt`

### Step 4: Set Environment Variables

#### Step 4.1 Configure Postgres Environment Variables
If you look into `utils/config.py`, you'll see that the database is hosted locally and that you need to specify its connection string through environment variables. Here's how to:

>```
>$ export DB_USER=<username>
>$ export DB_PASS=<password>
>$ export DB_ADDR=<localhost:5432>
>$ export DB_NAME=<fbo>
>```

Note that you do not need to type the `<` and `>` characters. You can always check the values of these environment variables with:

>`$ printenv`

#### Step 4.2 Configure Qualtrics Environment Variables
We also access our Qualtrics API token and survey_id as environment variables in `utils/config.py`. Set those like this:

>```
>$ export QUALTRICS_API_TOKEN=<token>
>$ export QUALTRICS_SW_SURVEY_ID=<site_wide_survey_id>
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



### Alternate: Docker & Docker Compose

#### Step 1 Install Docker

https://docs.docker.com/compose/install/

#### Step 2 Configure Environment Variables

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
    
>```
>$ docker-compose up
>```
  
(In another terminal)
>```
>$ docker exec --user hsm -it mlaas-web /bin/bash -c "python ~/HSM/app.py"
>```


 
## TODO
 - log performance of the models (based on the ground truth established by the validation)
 - include a training data table in the database. Move training data there and include support to insert validated samples there.
 - retrain the classifier if a certain threshold of newly validated samples is met
 - extend to the page-level survey
 - include unit and integration tests
