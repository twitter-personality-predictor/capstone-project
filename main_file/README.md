# Myers Briggs Twitter Personality Predictor
### By Joe Bennett, Morgan Cross, Michcael Haerle, Richard Macken

This project is designed to analyze the relationship between the 16 Myers Briggs personality types and twitter behavior. 
----

## Project Overview:

#### Objectives:
- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook Final Report.
- Create modules (wrangle.py, explore.py, model.py) that make the process repeateable and the report (notebook) easier to read and follow.
- Ask exploratory questions of the data that will help you understand more about the attributes. Answer questions through visualizations and statistical tests.
- Construct a model to predict Myers Briggs personality based on organic text in tweets posted on twitter.
- Refine work into a Report, in the form of a jupyter notebook, that our group will walk through in a 10 minute presentation to a group of collegues and instructors about the work we did, why, goals, what we found, our methdologies, and our conclusions.
- Be prepared to answer panel questions about our code, process, findings and key takeaways, and model.

#### Project Deliverables:
- this README.md walking through the project details
- final_report.ipynb displaying the process, findings, models, key takeaways, recommendation and conclusion
- wrangle.py with all data acquisition and preparation functions used
- model.py with all modeling functions 
- working_report.ipynb showing all work throughout the pipeline
- Canva slideshow to be used for our 10 minute presentation (https://www.canva.com/design/DAFQz8ntybg/qR2b9cpmcazG0GLN8FuhAA/view?utm_content=DAFQz8ntybg&      utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)  


----
## Executive Summary:

Goals:
- Analyze relationships between tweets and the myers briggs 16 personalities.
- Build a classification model to predict a personality type based on a user's tweets.

Key Findings:
- Exploring domains produces more distinct contrast between groups than looking at all 16 personability types individually.

Takeaways:
- Our best model looking at all 16 personalities, a multinomial Naive Bayes using TF-IDF, performed at 17.5% accuracy on the test dataset. This barely outperforms the baseline at 17.2%.
- There is likely higher accuracy when modeling to domain instead of the 16 types.

Recommendations:
- Due to the low accuracy, it is not recommended to use this for individual predictions.
- However, looking at how types and domains interact can provide business insights for targeted marketing and training for known personality types.


----
#### Data Dictionary

| Target | Type | Description |
| ---- | ---- | ---- |
| type | str | the 16 myers briggs personality types |

| Feature Name | Type | Description |
| ---- | ---- | ---- |
| name | str | the owner of the twitter account scraped |
| lemmatized | str | the lemmatized version of past 100 tweets scraped by user |
| personality_domain | str | which of the 4 domains the personality type is in |
| sentiment | float | the compound sentiment score |
| message_length | int | the count of characters in the combined 100 tweets after lemmatization |
| word_count | int | the count of words in the combined 100 tweets after lemmatization |
| i_e | int | i for introvert, e for extrovert |
| s_n | int | s for sensing, n for intuitive |
| f_t | int | f for feeling, t for thinking |
| p_j | int | p for perceiving, j for judging |

----
## 1. Planning
 - Create deliverables:
     - README
     - final_report.ipynb
     - working_report.ipynb
     - canva presentation
 - Bring over functional wrangle.py, explore.py, and model.py files
 - Acquire the data from the Kaggle and Twitter via acquire.py's web scraping. Save the data locally.
 - Prepare and split the data via the prepare.py functions
 - Explore the data and define hypothesis. Run the appropriate statistical tests in order to accept or reject each null hypothesis. Document findings and takeaways.
 - Model a baseline in predicting personality type and document the accuracy.
 - Fit and train classification models to predict personality type or domain on the train dataset.
 - Evaluate the models by comparing the train and validation data.
 - Select the best model and evaluate it on the train data.
 - Develop and document all findings, takeaways, recommendations and next steps.

 ----
## 2. Data Wrangling
This step calls the wrangle function from the wrangle.py. This function:
- Acquires the data and utilizes a pickle file to save it locally
- Cleans the data using the nltk tokenizer and lemmitizer
- Handles nulls
- Feature engineers:
    - personality_domain -> bins the 16 personalities to their domain
    - sentiment -> compound score using the Senitment Intensity Analyzer
    - message_length -> count of characters in each tweet after lemmitization
    - word_count -> count of words in each tweet after lemmitization
    - personality pairs -> creates boolean features for each pair
- Splits the data into 60/20/20 for train, validate, and test datasets


## 3. Data Exploration
This is where we take a look at the distribuion and shape our data. From there we evaluate our hypotheses and perform statistical validation testing.  Our questions and hypotheses are as follows:
### Question 1: What scale of groupings shows a significant relationship in sentiment? Pairs, types, or domains?
*    Ho -> The mean sentiment for explorers is less than or equal to the mean sentiment for analysts
*    Ha -> The mean sentiment for explorers is greater than the mean sentiment for analysts
     T-Test, Two-Sample, One-Tailed

>Fail to Reject the Null Hypothesis.
> Findings suggest there is less than or equal mean sentiment between analysts and sentinels.


### Question 2a: What words are seen across all personality types?
### Question 2b: What words are unique to introvert or extrovert?
*    Ho -> The mean tweet length is shorter or equal for extroverts than introverts
*    Ha -> The mean tweet length is longer for extroverts than introverts
     T-Test, Two-Sample, One-Tailed

> Fail to Reject the Null Hypothesis.
> Findings suggest the mean tweet length is shorter or equal for extroverts compared to introverts.


### Question 3: Is there a relationship between word count and personality type?

> - Across the board there are mostly consistent word counts for domains and the 16 personalities
> - INTJ is the one exception to this, and it is likely due to a lack of data for that type


---
## Data Modeling
The goal is to maximize accuracy.

| Features Dropped | Features Kept |
| ---- | ---- |
| pairs | lemmatized |
| domain | bow |
| sentiment | TF-IDF |
| message_length |  |
| word_count |  |
|  |  |

'Baseline Accuracy: 17.2'

Top Model per Feature
-------------
Bag of Words MultinomialNB Train Accuracy: 84.96%
-------------
Bag of Words MultinomialNB Validate Accuracy: 17.50%
-------------
TF-IDF MultinomialNB Train Accuracy: 25.63%
-------------
TF-IDF MultinomialNB Validate Accuracy: 17.50%
-------------


Top Clustering Model
-------------
Cluster MultinomialNB Train Accuracy: 19.50%
-------------
Cluster MultinomialNB Validate Accuracy: 15.00%
-------------
      
-------------
TF-IDF MultinomialNB Test Accuracy: 17.50%
-------------


### Modeling Takeaways:
- The best model was the TF-IDF Multinomial Naive Bayes model at % accuracy on test.
- The other models showed strong signs of overfitting the data on train, but were producing similar results on validate.


---
## Conclusion
Using a tweets to analyze behavior and classify into one of the 16 Myers Briggs personality types is not a reliable method. Our best model performed at 17.5% accuracy vs baseline's 17.2%. 

### Recommendations
- Due to the low accuracy, it is not recommended to use this for individual predictions
- However, looking at how types and domains interact can provide business insights for targeted marketing and training for known personality types

### Next Steps:
- Try two new approaches in modeling:
    - classifying based on domain
    - completing 4 regressions on each pair and concatenating the results
- Explore further associations in the words to inlcude emojis, swear words, ect. 
---

# How to reproduce our work
- Download this file https://drive.google.com/file/d/1k7Aq0P5hOPj1D3ndnwXDXkZ7tQzOYUrf/view?usp=share_link
  (This is to avoid the long web-scarping duration and repeatable results.)
- Download all files in the main repository.
- Open the final notebook and run top to bottom.