import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv(r"D:\Downloads\ReviewsAnalysis_NLP-master\restaurant-sentiment-analysis\Dataset\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
NaiveBayes = GaussianNB()
NaiveBayes.fit(X_train, y_train)
y_predNaiveBayes = NaiveBayes.predict(X_test)

from sklearn.metrics import confusion_matrix
cmNB = confusion_matrix(y_test, y_predNaiveBayes)

from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
RandomForest.fit(X_train, y_train)
y_predRandomForest = RandomForest.predict(X_test)
cmRF = confusion_matrix(y_test, y_predRandomForest)

from sklearn.model_selection import cross_val_score
accuracies_naivebayes = cross_val_score(NaiveBayes, X_train, y_train, cv=10)
accuracies_randomforest = cross_val_score(RandomForest, X_train, y_train, cv=10)

print("Naive Bayes Mean Accuracy:", accuracies_naivebayes.mean())
print("Naive Bayes Standard Deviation:", accuracies_naivebayes.std())
print("Random Forest Mean Accuracy:", accuracies_randomforest.mean())
print("Random Forest Standard Deviation:", accuracies_randomforest.std())

classifiers = ['Naive Bayes', 'Random Forest']
mean_accuracies = [accuracies_naivebayes.mean(), accuracies_randomforest.mean()]
std_deviations = [accuracies_naivebayes.std(), accuracies_randomforest.std()]

r1 = np.arange(len(mean_accuracies))
r2 = [x + 0.25 for x in r1]

plt.figure(figsize=(10, 6))
plt.bar(r1, mean_accuracies, width=0.25, edgecolor='grey', label='Mean Accuracy')
plt.bar(r2, std_deviations, width=0.25, edgecolor='grey', label='Standard Deviation')

plt.xlabel('Classifiers')
plt.xticks([r + 0.125 for r in range(len(mean_accuracies))], classifiers)
plt.ylabel('Scores')
plt.title('Mean Accuracy and Standard Deviation')
plt.legend()
plt.show()
