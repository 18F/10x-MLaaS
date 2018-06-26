# Survey Comment Classification:  Spam or Ham?
This folder contains two notebooks. In order for these to run, you'll need to
manually move the required assets into your local copy of the repo.

## Key Terms
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

## Limitations
  - Model training can be time consuming since hyper-parameter grid
  searching has combinatorial complexity.
    - A possible solution would be to substitute RandomizedSearchCV for
      GirdSearchCV
  - Supervised learning requires human involvement
    - So long as the training dataset remains small, someone will need to
    periodically review new predictions, hand correcting mis-classifications,
    and add to the training dataset before re-training the model.
    - This process could be facilitated by a GUI, such as one created with
    `tkinter`
  - The survey questions might benefit from revisions that narrow their scope
    - Some of the questions overlap while others are somewhat vague and yet
    another actually asks two questions. This greatly increases the scope of
    possible responses, which makes a machine learning classifier's job much
    more difficult (especially when the training dataset is small).

## Next Steps
  - Hand-label responses to the other two open-ended fields in the site-wide survey and then train/test models.
  - Expand to the page-level surveys
    - First, hand-label responses
    - Then try to use models created from the site-wide survey
    - Combine site-wide and page-level training data to create a new model
    and test on both the page- and site-level responses to see if there are improvements.
  - Once we have suitable models for all of the comment fields, explore the
  possibility of predicting the class for all four comments at once. This would
  involve using the class prediction as an input for the class prediction of another model.  
