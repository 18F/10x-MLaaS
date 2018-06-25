# Survey Comment Classification:  Spam or Ham?
This folder contains two notebooks. In order for these to run, you'll need to
manually move the required assets into your local copy of the repo.

## Definitions
  - **Spam**: Spam is defined broadly to mean any survey response that doesn't
    provide meaningful, relevant information.
  - **Ham**:  Ham is defined broadly to mean any survey response that provides
    meaningful, relevant information.
  - **The Site-Wide Survey**:  This survey has a 2% chance of triggering per
    page load for visitors. Since it is random, there is less trolling. Even so,
    approximately 1/3 of the comments are spam
  - **The Page-Level Survey**:  This survey appears at the bottom of most pages.
    Since it's always there, there tends to be more trolling and spam.
  - **The Comments**:  There are four fields that allow free-form responses:
    - `Please tell us what you value most about this website.`
    - `You answered "Other." Would you tell us a few words about the reason for
      your visit?`
    - `Please tell us more about the purpose of your visit or how we helped.`
    - `Please tell us why you were not able to fully complete the purpose of your
      visit today.`

## ModelTraining.ipynb
  - This notebook contains exploratory data analysis and model selection
  work. It's main output is a pickled sklearn model that can be used to predict
  on unseen data.

## CommentClassification.ipynb
  - This notebook unpickles a model identified in ModelTraining.ipynb and uses
  it to predict on new data.
