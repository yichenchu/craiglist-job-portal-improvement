import pandas as pd
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

df = pd.read_csv('job.csv')
print(df.shape)

df = df.drop_duplicates()
print(df.shape)


##Removing punctuations:
def punct(text):
    no_punct = [words for words in text if words not in string.punctuation]
    words = ''.join(no_punct)
    return words
df['exc_punct_body'] = df['body'].apply(lambda x: punct(x))

#Tokenizing
def tokenize(text):
    token = nltk.word_tokenize(text.lower())
    return token
df['token_body']=df['exc_punct_body'].apply(lambda x: tokenize(x))

#Lemmatizing
lemmatizer = nltk.stem.WordNetLemmatizer()
df['Lemmatized_body'] = df['token_body'].apply(lambda x :[lemmatizer.lemmatize(item) for item in x if item.isalpha()])

#Stop words Removal
stop = stopwords.words('english')
df['nostp_body'] = df['Lemmatized_body'].apply(lambda x:[item for item in x if item not in stop])
df['nostp_body'] = df['nostp_body'].apply(lambda x: " ".join(x))

#TF-IDF Vectorizer with uni and bigarms
vectorizer_body = TfidfVectorizer(ngram_range=(1,2), min_df=6)

##Topic modelling on the description column
vect = vectorizer_body.fit_transform(df['nostp_body'])
terms = vectorizer_body.get_feature_names_out()

from sklearn.decomposition import LatentDirichletAllocation
lda_body = LatentDirichletAllocation(n_components=5,max_iter=50,random_state=28).fit(vect)
topic_res = lda_body.transform(vect)
df['topic_body'] = topic_res.argmax(axis=1)

for topic_idx, topic in enumerate(lda_body.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[:-8-1:-1]]))
df['topic_body'].value_counts().plot.bar()


##Cleaning the Title column in the same way
df['exc_punct_title'] = df['Title'].apply(lambda x: punct(x))
df['token_title']=df['exc_punct_title'].apply(lambda x: tokenize(x))
df['Lemmatized_title'] = df['token_title'].apply(lambda x :[lemmatizer.lemmatize(item) for item in x if item.isalpha()])
df['nostp_title'] = df['Lemmatized_title'].apply(lambda x:[item for item in x if item not in stop])
df['nostp_title'] = df['nostp_title'].apply(lambda x: " ".join(x))

#TF-IDF Vectorizer for the title column
vectorizer_title = TfidfVectorizer(ngram_range=(1,2), min_df=3)

####Topic modelling on the title column
vect_title = vectorizer_title.fit_transform(df['nostp_title'])
terms_title = vectorizer_title.get_feature_names_out()

lda_title = LatentDirichletAllocation(n_components=3,max_iter=50,random_state=28).fit(vect_title)
topic_res_tit = lda_title.transform(vect_title)
df['topic_title'] = topic_res_tit.argmax(axis=1)

for topic_idx, topic in enumerate(lda_title.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([terms_title[i] for i in topic.argsort()[:-8-1:-1]]))
df['topic_title'].value_counts().plot.bar()


#Getting the target labels
Y = df['manual']
df_final = df.drop(['manual'],axis=1)
from sklearn.model_selection import train_test_split
#Splitting the train-test into 70, 30 ratio
X_train, X_test,Y_train, Y_test = train_test_split(df_final,Y,test_size = 0.3, random_state=28)

#Fit transform the tf-idf on to train data
vect_af = vectorizer_body.fit_transform(X_train['nostp_body'])
vect_fin = vectorizer_body.transform(X_train['nostp_body'])
vect_title_af = vectorizer_title.fit_transform(X_train['nostp_title'])
vect_title_fin = vectorizer_title.transform(X_train['nostp_title'])

#Transfrom using the train tf-idf vocabulary
vect_fin_test = vectorizer_body.transform(X_test['nostp_body'])
vect_title_fin_test = vectorizer_title.transform(X_test['nostp_title'])


##Modelling techniques
from scipy import sparse
X_train_final = sparse.hstack((vect_fin,vect_title_fin))
X_train_final = X_train_final.toarray()

X_test_final = sparse.hstack((vect_fin_test,vect_title_fin_test))
X_test_final = X_test_final.toarray()

##1st Iteration
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

SVM = SVC()
MNB = MultinomialNB()
GNB = GaussianNB()
XGB = xgb.XGBClassifier(use_label_encoder=False,random_state=28)
LR =LogisticRegression()
RF = RandomForestClassifier(n_estimators=30, random_state=14)
MLP = MLPClassifier(random_state=1)

models = {'SVM' : SVM,'MNB': MNB,'GNB' : GNB, 'XGB': XGB, 'LR': LR, 'RF': RF, 'MLP': MLP}

#Temporary Function to iteratively train the model
def train(cl, features, targets):
    cl.fit(features, targets)

#Temporary function to predict the output
def predict(cl, features):
    return (cl.predict(features))

#looping through all models
pred_scores = []
for k,v in models.items():
    train(v, X_train_final, Y_train)
    pred = predict(v, X_test_final)
    pred_scores.append((k, [accuracy_score(Y_test , pred)]))


#Hyper Parameter Tuning - XGB Classifier
model = xgb.XGBClassifier(use_label_encoder=False,random_state=14)
param = {'learning_rate': [0.001, 0.1, 1, 10]}
grid_xg = GridSearchCV(model, param, scoring='roc_auc', cv=10, return_train_score=True)
mod = grid_xg.fit(X_train_final, Y_train)
pred = mod.predict(X_test_final)
print(grid_xg.best_params_)


#Hyper Parameter Tuning - Logistic Regression
model = LogisticRegression()
param = {
    'penalty': ['l1','l2'],
    'solver': ['newton-cg','lbfgs','liblinear']}
clf = GridSearchCV(model, param, scoring='roc_auc', cv=10, return_train_score=True)
clf_log = clf.fit(X_train_final, Y_train)
y = clf_log.predict(X_test_final)
print(clf_log.best_params_)

#Hyper Parameter Tuning - MLP
model = MLPClassifier()
param = {
    'hidden_layer_sizes': [(5,3), (6,5), (10,3)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001,0.05,0.1],
    'learning_rate': ['constant','adaptive'],
}
clf_mlp = GridSearchCV(model, param, scoring='roc_auc', cv=10, return_train_score=True)
mod = clf_mlp.fit(X_train_final, Y_train)
y = mod.predict(X_test_final)
print(clf_mlp.best_params_)

##Getting ROC Scores and confusion matrices of the models

#Logistic Regression
model = LogisticRegression(solver='liblinear',penalty='l2')
mod = model.fit(X_train_final, Y_train)
y = mod.predict(X_test_final)
print(confusion_matrix(Y_test, y))
print(accuracy_score(Y_test, y))
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
display.plot()
plt.show()


#XGB Classifier
model = xgb.XGBClassifier(use_label_encoder=False,random_state=14,learning_rate=0.1)
mod = model.fit(X_train_final, Y_train)
y = mod.predict(X_test_final)
print(confusion_matrix(Y_test, y))
print(accuracy_score(Y_test, y))
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
display.plot()
plt.show()

#MLP Classifier
model = MLPClassifier(solver='adam', activation='relu',alpha=0.1,hidden_layer_sizes=(5,3), random_state=1,max_iter=30,verbose=10,learning_rate_init=0.1).fit(X_train_final, Y_train)
mod = model.fit(X_train_final, Y_train)
y = mod.predict(X_test_final)
print(confusion_matrix(Y_test, y))
print(accuracy_score(Y_test, y))
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
display.plot()
plt.show()