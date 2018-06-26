# Survey Comment Classification:  Spam or Ham?
This folder contains two notebooks. In order for these to run, you'll need to manually move the required assets into your local copy of the repo.

## Key Terms
  - **Spam**: Spam is defined broadly to mean any survey response that doesn't provide meaningful, relevant information.
  - **Ham**:  Ham is defined broadly to mean any survey response that provides meaningful, relevant information.
  - **The Site-Wide Survey**:  This survey has a 2% chance of triggering per page load for visitors. Since it is random, there is less trolling. Even so, approximately 1/3 of the comments are spam.
  - **The Page-Level Survey**:  This survey appears at the bottom of most pages.
    Since it's always there, there tends to be more trolling and spam.
  - **The Comments**:  There are four fields that allow free-form responses:
    - `Please tell us what you value most about this website.`
    - `You answered "Other." Would you tell us a few words about the reason for your visit?`
    - `Please tell us more about the purpose of your visit or how we helped.`
    - `Please tell us why you were not able to fully complete the purpose of your
      visit today.`

## ModelTraining - Value Comments.ipynb
  - This notebook contains exploratory data analysis and model selection work for the `Please tell us what you value most about this website.` question. 
  - It's main output is a pickled sklearn model that can be used to predict on unseen data.

## CommentClassification.ipynb
  - This notebook unpickles a model identified in ModelTraining - Value Comments.ipynb and uses it to predict on new data, correctly classifying 10/11 responses. 

## Limitations
  - So far, I've only looked at the site-wide data, which is fairly unbalanced (i.e. spam and ham is not 50/50). This tends to make the classifiers perform poorly in terms of spam recall (i.e. their ability to correctly classify a high percentage of the total spam). Even so, the models tend to be accurate overall and do a good job recalling all of the ham.
  - Model training can be time consuming since hyper-parameter grid searching has combinatorial complexity.
    - A possible solution would be to substitute RandomizedSearchCV for GirdSearchCV
  - Supervised learning requires human involvement
    - So long as the training dataset remains small, someone will need to periodically review new predictions, hand correcting mis-classifications, and add to the training dataset before re-training the model.
  - The survey questions might benefit from revisions that narrow their scope
    - Some of the questions overlap while others are somewhat vague and yet another actually asks two questions. This greatly increases the scope of possible responses, which makes a machine learning classifier's job much more difficult (especially when the training dataset is small).

## Next Steps
  - Hand-label responses to the other two open-ended fields in the site-wide survey and then train/test models.
  - Expand to the page-level surveys
    - First, hand-label responses
    - Then try to use models created from the site-wide survey
    - Combine site-wide and page-level training data to create a new model and test on both the page- and site-level responses to see if there are improvements.
  - Once we have suitable models for all of the comment fields, explore the possibility of predicting the class for all four comments at once. This would involve using the class prediction as an input for the class prediction of another model. 
  - Create Command Line version of this tool. Or use tkinter to create a simple GUI.

## Proposed Sustainable Process
In order for this project to outlive the term of my rotation, it needs to be easy to use. That means codifying the process and providing a simple means to execute. 
### The Process
*Assumption*: This assumes we've hand-labeled plenty of data and have trained suitable models for each comment field.

  1. Get the Data
      - Whether we get the data manually through the current tool or via an API doesn't matter at this point.
  2. Predict spam or ham for each comment field
  3. Review the classifications
      - Classification decisions can be sorted by the classifier's prediction probability. This helps the reviewer quickly review the edge cases instead of the near certainties.
  4. Correct mis-classifications
  5. Append the newly labeled data to the training dataset
      - It's very important that we only add correctly labeled data to the training dataset
      - There should be an easy way to review the hand-labled data (like opening an Excel file)
  6. Optionally re-train the model on the now larger training dataset
     - This should probably only happen once the training dataset has grown by a certain percentage 
     - Need to be careful about compute and time resources here!
  7. Ideally, apply some analytics options to the ham comments
     - Latent Dirichlet Allocation (LDA) and/or Non-negative Matrix Factorization (NMF) for topic modeling
     - Create a tool that helps the analyst find comments similar to one they have in hand. This comment they have in hand could be a real one, or something they make up akin to a search.
