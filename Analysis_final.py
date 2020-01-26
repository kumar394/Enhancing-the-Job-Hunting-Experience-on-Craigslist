import pandas as pd
import urllib
import nltk
from nltk.corpus import stopwords
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import *
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

import PyPDF2



#####Build Model for Standardizing PayType#####

#Take data of Compensation Type
df_model = pd.read_excel('Compensation Dataset.xlsx')


model_compensation = df_model['Description'].values.tolist()
model_label = df_model['Compensation'].values.tolist()


lemmatizer= nltk.stem.WordNetLemmatizer()
processed_compensation=[]

#Dimension Reduction of Compensation Type
for doc in model_compensation:
    tokens = nltk.word_tokenize(str(doc).lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    processed_compensation.append(" ".join(tokens))


vectorizer = TfidfVectorizer(ngram_range = (1,2))
vectorizer.fit(processed_compensation)
v = vectorizer.transform(processed_compensation)

#Splitting the Data
x_train = v[0:3001].toarray()
x_test = v[3001:3704].toarray()
y_train = model_label[0:3001]
y_test = model_label[3001:3704]

#Training and Testing with different ML

#Naive Bayes
NBmodel = MultinomialNB()
NBmodel.fit(x_train, y_train)
y_pred_NB = NBmodel.predict(x_test)

#Logistic
LGmodel = LogisticRegression()
LGmodel.fit(x_train, y_train)
y_pred_LG = LGmodel.predict(x_test)

#SVM
SVMmodel = LinearSVC()
SVMmodel.fit(x_train, y_train)
y_pred_SVM = SVMmodel.predict(x_test)

#RandomForest
RFmodel = RandomForestClassifier(n_estimators=500, max_depth=3,bootstrap=True, random_state=0)
RFmodel.fit(x_train, y_train)
y_pred_RF = RFmodel.predict(x_test)

#Deep Learning
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4), random_state=1)
DLmodel.fit(x_train, y_train)
y_pred_DL= DLmodel.predict(x_test)

#Checking Accuracy Score of all the models
acc_NB = accuracy_score(y_test, y_pred_NB)
print("Naive Bayes model Accuracy: {:.2f}%".format(acc_NB*100))
acc_LG = accuracy_score(y_test, y_pred_LG)
print("Logistic Regression model Accuracy: {:.2f}%".format(acc_LG*100))
acc_SVM = accuracy_score(y_test, y_pred_SVM)
print("SVM model Accuracy: {:.2f}%".format(acc_SVM*100))
acc_RF = accuracy_score(y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))
acc_DL = accuracy_score(y_test, y_pred_DL)
print("DL model Accuracy: {:.2f}%".format(acc_DL*100))


#####Data Preparation#####

df = pd.read_csv('Jobs_Final.csv', index_col = 0)

description = df['Description'].values.tolist()
title = df['Title'].values.tolist()
compensation = df['Compensation'].values.tolist()
jobType = df['Type'].values.tolist()

merged = []
for i in range(len(title)):
     merged.append(title[i] + " " + description[i])

lemmatizer= nltk.stem.WordNetLemmatizer()
tokenizer = nltk.tokenize.WhitespaceTokenizer()

#Extract only noun from the Description and Title of the Job
processed_description = []
i = 0
for doc in merged:
    token_doc = nltk.word_tokenize(doc.lower())
    tokens = [lemmatizer.lemmatize(token) for token in token_doc if token.isalpha()]
    POS_token_doc = nltk.pos_tag(tokens)
    POS_token_temp = []
    for i in POS_token_doc:
        if i[1] == 'NN':
            POS_token_temp.append(i[0])
    processed_description.append(" ".join(POS_token_temp))


processed_compensation = []

#Dimension Reduction of Processed Job Information
for doc in compensation:
    tokens = nltk.word_tokenize(str(doc).lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    processed_compensation.append(" ".join(tokens))



######RELEVANCE JOB SEARCH GROUPING#######
#Distribute Job Listing into 31 categories
vectorizer_topic = CountVectorizer(ngram_range = (1,2), min_df = 5)
vectorizer_topic.fit(processed_description)

v_topic = vectorizer_topic.transform(processed_description)
terms = vectorizer_topic.get_feature_names()

lda = LatentDirichletAllocation(n_components=31).fit(v_topic)
topicnames = ["Topic: " + str(i) for i in range(1, len(lda.components_) + 1)]

doc_topic = lda.transform(v_topic)

dominant_topic = (np.argmax(doc_topic, axis=1) + 1).tolist()


########JOB SEARCH#######
import PyPDF2

#Uploading RESUME
# pdfFileObject = open('1.pdf', 'rb')
# pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
# count = pdfReader.numPages
# search_term = ""
# for i in range(count):
#     page = pdfReader.getPage(i)
#     search_term += page.extractText()
#
# search_term = search_term.replace('\n', '')
search_term = "Truck Driver"
pay_type = "Hourly"

vectorizer_tfidf = TfidfVectorizer(ngram_range = (1,1))
vectorizer_tfidf.fit(processed_description)

v2 = vectorizer_tfidf.transform(processed_description).toarray()
sample = vectorizer_tfidf.transform([search_term]).toarray()

score = []
#Similarity Score between Job search term or Resume and Job Description and Title
for i in range(len(processed_description)):
    if processed_description[i] != '':
        number = cosine_similarity([v2[i,:]], [sample[0,:]])[0][0]
        score.append(number)

#Ranking the jobs according to their similarity score
rank = np.argsort(score)[::-1][:len(score) + 1].tolist()

result_title = []
result_description = []
result_compensation = []
result_jobType = []
result_revelanceJob = []

for i in rank:
    # Iterate until the similarity score is greater than 0
    if score[i] > 0:

        tokens = nltk.word_tokenize(str(processed_compensation[i]).lower())
        if len(tokens) == 0:
            tokens.append("")
        type_test = vectorizer.transform(tokens).toarray()

        # Predicting the Compensation Type of the Job
        pred_type = SVMmodel.predict(type_test)
        pred_type = pred_type.item(0)

        # Selecting the job only if Pay type predicted matches the required Pay Type
        if pred_type == pay_type:
            # Appending the Relevance Job ID number
            rel = ""

            # Matching the Relevant Job for a particular Job, the Pay/ Compensation Type of the Job should match as well
            for k in range(len(dominant_topic)):
                if dominant_topic[k] == dominant_topic[i] and i != k:
                    relevanceSimilarity = cosine_similarity([v2[i, :]], [v2[k, :]])[0][0]

                    if relevanceSimilarity > 0.2:
                        tokens = nltk.word_tokenize(str(processed_compensation[k]).lower())
                        if len(tokens) == 0:
                            tokens.append("")
                        type_test = vectorizer.transform(tokens).toarray()

                        if SVMmodel.predict(type_test).item(0) == pay_type:
                            if len(rel) == 0:
                                rel += str(k)
                            else:
                                rel = rel + ',' + str(k)

            result_title.append(title[i])
            result_description.append(description[i])
            result_compensation.append(compensation[i])
            result_jobType.append(jobType[i])
            result_revelanceJob.append(rel)

df = pd.DataFrame(list(zip(result_title, result_description, result_compensation, result_jobType, result_revelanceJob)),
               columns =['Title', 'Description', 'Compensation', 'JobType', 'RelevanceJob'])

#Export to CSV
df.to_csv('Jobs_ShortList.csv', index=True)




