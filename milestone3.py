import pandas as pd
import numpy as np

### Data Preprocessing ###
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
#sentiment analysis
from textblob import TextBlob 
#text processing
import re
import nltk

### Hyperparameter tuning ###
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

### models ###
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

### Evaluation ###
from sklearn.metrics import (
    accuracy_score,
    classification_report 
)

####### READING IN DATA & PREPROCESSING #######
dtypes = {
    "Unnamed: 0": "numerical",
    "urlDrugName": "categorical",
    "rating": "numerical",
    "effectiveness": "categorical",
    "sideEffects": "categorical",
    "condition": "categorical",
    "benefitsReview": "text",
    "sideEffectsReview": "text",
    "commentsReview": "text",
}

#Load training data
train_data = pd.read_csv('drugLib_raw/drugLibTrain_raw.tsv', sep='\t')
#Load test data
test_data = pd.read_csv('drugLib_raw/drugLibTest_raw.tsv', sep='\t')


### Review Preprocessing (for text values) ###
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

### Clean for sentiment analysis ###
train_data["sideEffectsReview"] = train_data["sideEffectsReview"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True))
train_data["commentsReview"] = train_data["commentsReview"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True))
train_data["benefitsReview"] = train_data["benefitsReview"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True))


train_data['sideEffectsReview'] = train_data['sideEffectsReview'].astype(str)
train_data["sideEffectsSentiment"] = train_data["sideEffectsReview"].apply(lambda x: TextBlob(x).sentiment.polarity)

train_data['commentsReview'] = train_data['commentsReview'].astype(str)    
train_data["commentsSentiment"] = train_data["commentsReview"].apply(lambda x: TextBlob(x).sentiment.polarity)


train_data['benefitsReview'] = train_data['benefitsReview'].astype(str)    
train_data["benefitsSentiment"] = train_data["benefitsReview"].apply(lambda x: TextBlob(x).sentiment.polarity)

################## 

test_data["sideEffectsReview"] = test_data["sideEffectsReview"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True))
test_data["commentsReview"] = test_data["commentsReview"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True))
test_data["benefitsReview"] = test_data["benefitsReview"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True))


test_data['sideEffectsReview'] = test_data['sideEffectsReview'].astype(str)
test_data["sideEffectsSentiment"] = test_data["sideEffectsReview"].apply(lambda x: TextBlob(x).sentiment.polarity)

test_data['commentsReview'] = test_data['commentsReview'].astype(str)    
test_data["commentsSentiment"] = test_data["commentsReview"].apply(lambda x: TextBlob(x).sentiment.polarity)


test_data['benefitsReview'] = test_data['benefitsReview'].astype(str)    
test_data["benefitsSentiment"] = test_data["benefitsReview"].apply(lambda x: TextBlob(x).sentiment.polarity)
#########################################################

#overall sentiment classificaiton
train_data["avgSentiment"] = (train_data["sideEffectsSentiment"]+train_data["sideEffectsSentiment"]+train_data["sideEffectsSentiment"]) / 3
train_data["avgSentiment"] = round(train_data["avgSentiment"])

test_data["avgSentiment"] = (test_data["sideEffectsSentiment"]+test_data["sideEffectsSentiment"]+test_data["sideEffectsSentiment"]) / 3
test_data["avgSentiment"] = round(test_data["avgSentiment"])

## Handling the missing values and assigning old column names
train_imp, test_imp = [
    pd.DataFrame(
        SimpleImputer(strategy="most_frequent").fit_transform(df), columns=df.columns
    )
    for df in (train_data, test_data)
]

## Encoding the categorical columns
for i in ["urlDrugName", "condition", "sideEffects"]:
    train_imp[i] = LabelEncoder().fit_transform(train_imp[i])
    test_imp[i] = LabelEncoder().fit_transform(test_imp[i])

### drop text columns
train_imp.drop("benefitsReview", axis=1, inplace=True)
train_imp.drop("sideEffectsReview", axis=1, inplace=True)
train_imp.drop("commentsReview", axis=1, inplace=True)

test_imp.drop("benefitsReview", axis=1, inplace=True)
test_imp.drop("sideEffectsReview", axis=1, inplace=True)
test_imp.drop("commentsReview", axis=1, inplace=True)

train_imp.columns = train_imp.columns.astype(str)
test_imp.columns = test_imp.columns.astype(str)

#determine target based off of effectiveness and sentiment
#Design decision: if highly effective or considerably effective and has
#positive sentiment or neutral sentiment present -> effective classification
#else -> ineffective classification

targetTrain = (((train_imp["effectiveness"] == "Highly Effective") |
                       (train_imp["effectiveness"] == "Considerably Effective")) &
                       (train_imp["avgSentiment"] >= 0))

targetTest = (((test_imp["effectiveness"] == "Highly Effective") |
                       (test_imp["effectiveness"] == "Considerably Effective")) &
                       (test_imp["avgSentiment"] >= 0))

train_imp["target"] = (targetTrain).astype(int)
test_imp["target"] = (targetTest).astype(int)


## Splitting the train and test datasets into feature variables
X_train, Y_train = train_imp.drop(["effectiveness", "target"], axis=1), train_imp["target"]
X_test, Y_test = test_imp.drop(["effectiveness", "target"], axis=1), test_imp["target"]
###############################################################################

####### Models (Fitting, Tuning, Evaluation) #######

def print_metrics(y_true, y_pred):
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    
    # Classification Report
    class_report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(class_report)


####### SVM #######
svm_clf = svm.SVC()

param_grid = {'C': [0.1, 1, 2], 
              'gamma': [1, 0.1, 0.01],
              'kernel': ['rbf']} 


grid = GridSearchCV(svm_clf, param_grid, refit = True, verbose = 3, n_jobs=-1)

grid.fit(X_train,Y_train)

best_model = grid.best_estimator_

svm_y_pred = best_model.predict(X_test)

print("SVM Classifier Metrics:")
print_metrics(Y_test, svm_y_pred)

####### GNB #######
gnb = GaussianNB()

#Apply smoothing param
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(estimator=gnb, 
                 param_grid=params_NB, 
                 cv=3,
                 verbose=1, 
                 scoring='accuracy')

gs_NB.fit(X_train,Y_train)

best_params = gs_NB.best_params_

best_model = gs_NB.best_estimator_

gnb_y_pred = best_model.predict(X_test)
print("GNB Classifier Metrics:")
print_metrics(Y_test, gnb_y_pred)

####### Random Forest #######
param_grid = {
    'bootstrap': [True],
    'max_depth': [30, 40, 50, 100],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [350, 375, 400, 425, 450]
}

rf = RandomForestClassifier()

#Tune Hyperparameters
rand_rf = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, verbose=2, n_jobs = -1)

rand_rf.fit(X_train, Y_train)
best_params = rand_rf.best_params_
best_model = rand_rf.best_estimator_

rf_y_pred = best_model.predict(X_test)

print("Random Forest Classifier Metrics:")
print_metrics(Y_test, rf_y_pred)
###############################################################################




