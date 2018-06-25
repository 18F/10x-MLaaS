# Survey Comment Classification:  Spam or Ham?

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

## Classifying New Comments
  1. Visit https://admin-survey.usa.gov/ to get the most recent responses.
    - Choose the site-wide survey:  `USA.gov Customer Satisfaction Survey`
    - Choose to receive the csv format emailed to you.
  2. Download the responses that have been emailed to you.
  3. Move the file (e.g. `201806121828-USA.gov_Cus-1.1.csv`) to the `unlabeled_data` directory.
  4. Open `ModelTraining.ipynb` and follow the steps therein

## Training the Comment Classifier
  1. Open `CommentClassification.ipynb`. You can find documentation there.
