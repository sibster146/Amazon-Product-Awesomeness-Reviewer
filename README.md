# Amazon-Product-Awesomeness-Reviewer
CS 349 Machine Learning
Final Project Submission
Full documentation can be found here: https://drive.google.com/file/d/1M6T6x1b5ouzvEnf9Z_tALCW_H_-h7U9W/view?usp=sharing

**PROJECT DESCRIPTION**

The Amazon Product Awesomeness Reviewer is a machine learning model that determines if an Amazon product is awesome or not based on a set of reviews of that product. 
The raw features of the training set: asin (the id of the product reviewed), reviewerID, unixReviewTime, vote, verified, reviewerName, reviewText, summary, and awesomeness (the target value and determines if the product under review is awesome or not. From there, extra features were engineered such as reviewText sentiments and summary sentiments. 

Then, several models were tested with the SciKit Learn library. The best performing model was the Decision Tree with a bagging additive. 
