# Sentiment-Analysis

## Title: 

What’s the vibe? Twitter Sentiment Analysis Final Project

## Who: 

James Pirozzolo: jpirozzo

William Kantaros: wkantar1

Julian Croonenberghs: jcroonen

## Introduction: What problem are you trying to solve and why?
### Problem: 

In particular, because of the recent election and current political climate, we believe it is more necessary than ever to be able to predict the sentiment surrounding a particular topic or message, and Twitter - a product built on rapid, opinionated interactions by users - is a great tool for that. Additionally, quick sentiment changes can often have major effects on the stock market, polls, brand favorability, etc. 

### Solution: 

We believe that building a sentiment analysis deep learning network that trains on Tweets may be particularly helpful. Since Twitter is a platform essentially built on relaying opinions (and many of them, we might add), we believe that this is a great platform to study. While this isn’t a new idea, we are looking to build a new model from scratch based on the tools we have learned from this course.  

### What is sentiment analysis?: 

the computational analysis of spoken or written (in our case written) language, and the subsequent classification of this language as either: positive, negative, or neutral, as well as the degree of the sentiment. 

## Related Work: Are you aware of any, or is there any prior work that you drew on to do your project?

We found an interesting article that discusses a similar concept relating to sentiment analysis on Tweets. This one in particular focuses on Netflix reviews, and categorizing them as positive, negative or neutral. Similar to what we plan to do, this model uses LSTMs, but also discusses RNNs and how they can equally be used in this implementation. The article also discusses preprocessing, which deals with rearranging the data (phrases) in order for the model to train more easily. However, in general for this type of analysis, preprocessing is not necessarily very complex. For example, it includes removing special characters and word indexing.

### Some urls: 

- https://towardsdatascience.com/sentiment-analysis-with-deep-learning-62d4d0166ef6
- http://help.sentiment140.com/for-students

## Data: What data are you using (if any)?

### The dataset: 

We found a dataset called “Sentiment140”, which is composed of tweets, as well as fields for the ID, sentiment label, date and time, author, and text of each tweet. We feel that this dataset is incredibly valuable and pertinent to our topic, and labels each Tweet in the database on a 0-4 scale for how favorable/unfavorable it is. Another option would be to build the dataset ourselves by loading tweets directly from the internet.
### Accessing it: 

CSV file format, which we will then preprocess by extracting the data and formatting it appropriately (very similar to our NLP homework assignments). 

### Methodology: 
What is the architecture of your model?

Recurrent Neural Networks (RNNs), which are ideal for dealing with sequential information such as text, are an interesting deep learning architecture to implement for this project. Particularly, when it comes to sentiment analysis, we want to use an RNN type architecture instead of a “traditional” feed-forward network because we are looking to learn the meaning/sentiment of a piece of text, in which case we must be able to keep track of phrase structure, and the order of the words. However, we are thinking of using a Long Short-Term Memory (LSTM) network, which is technically a special kind of RNN, but that would allow for us to store memory in an additional way, using cell states, in addition to the hidden states. This feature also helps enable LSTMs to deal with much longer sequences than RNNs. Since tweets are a maximum of 280 character, we think the difference wouldn’t be incredibly large, but believe an LSTM based model will be slightly more accurate. 

We will begin by implementing an RNN structure. Once we feel confident with that, we will move on to attempting an LSTM model. 
 
 
## Metrics: What constitutes “success?”

### Base Goal: 

Build a deep learning network that takes in tweets and evaluates their sentiment, regardless of the accuracy. 

### Target Goal: 

Build a deep learning network that takes in tweets and evaluates their sentiment with a reasonable accuracy 
### Stretch Goal: 

Build a website that would stream in tweets and print out their sentiment.

## Ethics: 
Choose 2 of the following bullet points to discuss; not all questions will be relevant to all projects so try to pick questions where there’s interesting engagement with your project. (Remember that there’s not necessarily an ethical/unethical binary; rather, we want to encourage you to think critically about your problem setup.)

### What broader societal issues are relevant to your chosen problem space?

Any discussion of twitter or any other online social media brings in the issue of fake news, as discussed in the most recent conceptual questions assignment. That being said, we think that the more prevalent issue is that of prejudice on the internet. For example, it is very possible that our model trains to assign labels (a particular sentiment) to posts about racism or other similar issues with a great deal of bias due to the way that people talk about it online. Political posts, regardless of what they are, seem to always have an opposite on the other side of the aisle, and our sentiment analysis may discover biases of people who post online.

### What is your dataset? 
Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?

Our dataset is sentiment140, a set containing tweets and other relevant data for our computations (discussed above). Sentiment140 was created by four graduate students at Stanford for use in sentiment analysis. We have not found any concerns, nor have any ourselves, and due to its size alone, 1.6 million tweets, we believe that it is representative of the twitter ecosystem. It might contain any number of historical or societal biases. Notably, it may have particular biases regarding social issues as the tweets are a bit aged (2016 and earlier, I believe), but we hope that the biases do not have a great impact on the sentiment calculation.
How are you planning to quantify or measure error or success? What implications does your quantification have?
The dataset provides us with labels (sentiments that they calculated using multiple models). As a result, we are able to quantify the success of our model by comparing our results to that of the datasets expected sentiment. The quantification will determine the extent to which we stand behind the sentiment of our model. Additionally, matching the model too well may be a consequence of overfitting the data, which we hope to avoid (maybe using a dropout or similar feature).
## Division of labor: 

We believe that we work much better as a team, and will thus generally be working on most aspects of the project together, while splitting much smaller portions of the project as we go. For this reason, it is hard to concretely divide the workload before attacking the project head on. 
