# Imports
import pandas as pd
import numpy as np
# from flair.models import TextClassifier
# from flair.data import Sentence

# Importing of various classification tools that were tested
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


# Used classification tools
from sklearn.model_selection import cross_val_score

# To save Sentiment Analysis
import gzip
import pickle

def create_feature_vector(training_data,scores):
    # Preprocessing & Feature Generation
    training_data["format"] = training_data["style"].apply(lambda x: x["Format:"] if pd.notna(x) else None)
    # New Features
    training_data["review_length"] = training_data["reviewText"].apply(lambda text: len(text) if text != None else 0)
    training_data["vote"] = training_data["vote"].apply(lambda vote: 0 if pd.isna(vote) else int(vote.replace(',', '')) if isinstance(vote, str) else int(vote))
    # Reviewer Ratio Generation
    reviewers = training_data[['reviewerID', 'verified']].copy()
    reviewers['verification_ratio'] = reviewers['verified'].map(int)
    reviewers = reviewers.drop('verified', axis=1)
    reviewers = reviewers.groupby(['reviewerID']).mean()
    training_data = pd.merge(training_data, reviewers, on='reviewerID', how='left')

    summary_stats = pd.DataFrame()
    summary_stats['raw'] = training_data[['asin', 'summary_sentiment']].groupby('asin')['summary_sentiment'].apply(list)
    summary_stats['summary_sentiment_avg'] = summary_stats.raw.apply(lambda x: np.mean(x))
    summary_stats['summary_sentiment_std'] = summary_stats.raw.apply(lambda x: np.std(x))

    summary_stats['number_of_reviews'] = summary_stats.raw.apply(lambda list: pd.Series(list).count())

    text_stats = pd.DataFrame()
    text_stats['raw'] = training_data[['asin', 'text_sentiment']].groupby('asin')['text_sentiment'].apply(list)
    text_stats['text_sentiment_avg'] = text_stats.raw.apply(lambda x: np.mean(x))
    text_stats['text_sentiment_std'] = text_stats.raw.apply(lambda x: np.std(x))

    vote_weighted_stats = pd.DataFrame(training_data['asin'])
    vote_weighted_stats['vote_weighted_summary_sentiment'] = training_data['summary_sentiment'] * training_data['vote']
    vote_weighted_stats['vote_weighted_text_sentiment'] = training_data['text_sentiment'] * training_data['vote']
    vote_weighted_stats = vote_weighted_stats.groupby('asin').mean()

    verified_weighted_stats = pd.DataFrame(training_data['asin'])
    verified_weighted_stats['verified_text_sentiment'] = training_data['verified'] * training_data['text_sentiment']
    verified_weighted_stats['verified_summary_sentiment'] = training_data['verified'] * training_data['summary_sentiment']

    verified_weighted_stats['verified_vote_weighted_text_sentiment'] = training_data['verified'] * training_data['text_sentiment'] * training_data['vote']
    verified_weighted_stats['verified_vote_weighted_summary_sentiment'] = training_data['verified'] * training_data['summary_sentiment'] * training_data['vote']
    verified_weighted_stats = verified_weighted_stats.groupby('asin').mean()

    time_weighted_stats = pd.DataFrame(training_data['asin'])
    training_data['normedReviewTime'] = (training_data['unixReviewTime'] - training_data['unixReviewTime'].mean()) / training_data['unixReviewTime'].std()
    time_weighted_stats['avgReviewTime'] = training_data['normedReviewTime']

    product_format = training_data[['asin', 'format']]
    product_format.drop_duplicates(subset=['asin'])
    format_list = product_format.format.unique()
    formats = {}
    for i in range(len(format_list)):
        formats[format_list[i]] = i
    product_format.format = product_format.format.map(formats)
    product_format = product_format.drop_duplicates(subset='asin')

    aggregatedProductFeatures = pd.merge(summary_stats.drop('raw', axis=1), text_stats.drop('raw', axis=1), on='asin')
    aggregatedProductFeatures = aggregatedProductFeatures.merge(vote_weighted_stats, on='asin').merge(verified_weighted_stats, on='asin').merge(time_weighted_stats, on='asin')

    aggregatedProductFeatures = aggregatedProductFeatures.merge(product_format, on='asin')
    verified_review_ratio = training_data[['asin', 'verified']].groupby('asin').mean()
    aggregatedProductFeatures = aggregatedProductFeatures.merge(verified_review_ratio, on='asin')
    aggregatedProductFeatures['text_sentiment_std'] = aggregatedProductFeatures['text_sentiment_std'].apply(lambda std: 0 if pd.isna(std) else std)
    aggregatedProductFeatures['summary_sentiment_std'] = aggregatedProductFeatures['summary_sentiment_std'].apply(lambda std: 0 if pd.isna(std) else std)
    aggregatedProductFeatures = aggregatedProductFeatures.merge(scores, on='asin')
    return aggregatedProductFeatures

print("importing")
training_data = pd.read_pickle(r'C:\Users\sibys\OneDrive\Documents\school\cs349\CS349-main\project\source\training_data_uncompressed.pkl')
scores = pd.read_json(r'C:\Users\sibys\OneDrive\Documents\school\cs349\CS349-main\project\source\product_training.json')
test_3_data = pd.read_pickle(r'C:\Users\sibys\OneDrive\Documents\school\cs349\CS349-main\project\source\test_3_data.pkl')
test_3_scores = pd.read_json(r'C:\Users\sibys\OneDrive\Documents\school\cs349\CS349-main\project\source\test_3_scores.json')
print("imported training data and test 3 data")

aggregatedProductFeatures = create_feature_vector(training_data,scores)
X = aggregatedProductFeatures.drop(['asin', 'awesomeness'], axis=1)
y = aggregatedProductFeatures['awesomeness']

features = create_feature_vector(test_3_data , test_3_scores)
test_3_X = features.drop(['asin'], axis=1)
print("made feature vectors for training data and test 3 data")

print("fitting best model to training data")
best_clf = BaggingClassifier(DecisionTreeClassifier(max_depth = 10, max_features = .7, min_samples_leaf = 5, min_samples_split = 10), max_samples = 0.25, max_features = 1.0, n_estimators = 100)
best_clf.fit(X,y)

print("exporting model to external pickle file")
with open(r'C:\Users\sibys\OneDrive\Documents\school\cs349\CS349-main\project\source\best_clf.pkl', 'wb') as f:
    pickle.dump(best_clf, f)

print("importing model into program")
with open(r'C:\Users\sibys\OneDrive\Documents\school\cs349\CS349-main\project\source\best_clf.pkl', 'rb') as f:
    best_clf = pickle.load(f)

print("testing F1 of model to training data")
f1_scores = cross_val_score(best_clf, X, y, cv=10, scoring="f1", n_jobs=-1, verbose = 1)
print(f"Model's F1 score: {np.mean(f1_scores)}")

print("making predictions of test 3 with model")
features['review_predictions'] = best_clf.predict(test_3_X)
predictions = features.groupby('asin')['review_predictions'].mean().round()
predictions_df = predictions.reset_index()
predictions_df.columns = ['asin', 'awesomeness']
predictions_df['awesomeness'] = predictions_df['awesomeness'].astype(int)
predictions_df.to_json(r'C:\Users\sibys\OneDrive\Documents\school\cs349\CS349-main\project\predictions.json', orient='records')
print("predictions.json exported")