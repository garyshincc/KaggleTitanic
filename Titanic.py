# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# the python pandas package help us with datasets
# we read in the datasets into the Pandas DataFrames
# we also combine these datasets to run certain operations
# on both datasets together.
train_df = pd.read_csv('Titanic_Data/train.csv')
test_df = pd.read_csv('Titanic_Data/test.csv')
combine = [train_df, test_df]


''' data analysis '''
'''
we get ['PassengerId' 'Survived' 'Pclass' 
'Name' 'Sex' 'Age' 'SibSp' 'Parch'
'Ticket' 'Fare' 'Cabin' 'Embarked']

categorical: survived, sex, embarked (which port)
ordinal: pclass

continuous: age, fare
discrete: sibSp (num siblings onboard), Parch (num parents on board)

numeric and alphanumeric data type: tickets
alphanumeric: cabin

name feature - can containt errors or typose
also there are several ways to describe a name,
including titles, brackets, quotes, etc.
'''

# preview the data
print train_df.head()

# end of the data
print train_df.tail()

# what is the data types for these various features?
print ('_'*15 + "train_df.info()" + '_'*15)
print train_df.info()
print ('_'*15 + "test_df.info()" + '_'*15)
print (test_df.info())

print ('_'*15 + "train_df.describe()" + '_'*15)
print (train_df.describe())

'''

assumptions based on data analysis

Correlation - 

correlating
- how much of each feature correlate with Survival?
- Do early in the analysis, and compare with modelled ones later

completing
- we may want to complete age feature
	- age is definitely correlated to survival
- we may want to complete embareked feature as well
	- may also correlate with survival, OR
	- it may correlate with other feature

correcting
- ticket feature may be dropped from our analysis because...
	- contains high ratio of duplicates
	- dont see a correlation between ticket and survival
- Cabin feature can be dropped from our analysis because...
	- highly incomplete values (lots of null values)
- passegerID can be dropped from our analysis because...
	- does not contribute to survival
- Name feature may be dropped from our analysis because...
	- does not directly contribute to survival

creating
	Then we will create new features, in 3 ways :
	Simplifications of existing features
	Combinations of existing features
	Polynomials on the top 10 existing features

- May want to create a new feature based on parent count and sibling count
	- get total family count
- May want to engineer the Name feature to extract Title as a new features
	- such as doctor, lawyer, captain, etc...
- May want to create new feature for age bands.
	- turns this continuous numerical feature into categorical

classifying
- We may also add to our assumptions based on the problem description noted earlier
	- Women were more likely to have survived.
	- Children were more likely to have survived.
	- The upper class were more likely to have survived.

'''


# analysis by pivoting features
print ('_'*15 + "train_df pclass and survived" + '_'*15)
print (train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print ('_'*15 + "train_df groupby, survived" + '_'*15)
print (train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print ('_'*15 + "train_df sex, survived" + '_'*15)
print (train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print ('_'*15 + "train_df siblings/spouse, survived" + '_'*15)
print (train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print ('_'*15 + "train_df parent/children, survived" + '_'*15)
print (train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# analysis by visualizing data

# sns is seaborn package
# graph = sns.FacetGrid(train_df, col="Survived")
# graph.map(plt.hist, 'Age', bins=20)
# sns.plt.show()


# grid = sns.FacetGrid(train_df, col="Survived",
# 	row="Pclass", size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
# grid.add_legend();
# sns.plt.show()

''' we observe that ...

Female passengers had much better survival rate than males. 
	- Confirms classifying (Women were more likely to have survived).
Exception in Embarked=C where males had higher survival rate.
	- This could be a correlation between Pclass and Embarked
		and in turn Pclass and Survived,
		not necessarily direct correlation between
		Embarked and Survived.
Males had better survival rate in Pclass=3 when
compared with Pclass=2 for C and Q ports. 
	- Completing (we may want to complete embareked feature as well).
Ports of embarkation have varying survival rates for Pclass=3
and among male passengers.
	- Correlating (ticket feature may be dropped from our analysis).
'''

''' we can make a decision that ...
We should add "Sex" feature to model training.
We should complete and add Embarked feature to model training.
'''

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
#sns.plt.show()

# correlating categorical and numerical features

''' we observe that...
Higher fare paying passengers had better survival.
	- Confirms our assumption for creating (#4) fare ranges.
Port of embarkation correlates with survival rates.
	- Confirms correlating (#1) and completing (#2).
'''

''' we can make a decision that ...
We should consider banding the Fare feature.
'''
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
#sns.plt.show()

# Wrangling Data
# Execute our decisions and assumptions by correcting, creating
# and completing goals.

# recall combine = [train_df, test_df]
print ("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

combine = [train_df, test_df]

print ("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



# Creating new feature, extracting from existing

# we want to analyze title to correlation
''' we can observe that ...
Most titles band Age groups accurately.
	For example: Master title has Age mean of 5 years.

Survival among Title Age bands varies slightly.

Certain titles mostly survived (Mme, Lady, Sir)
or did not (Don, Rev, Jonkheer).
'''

''' we can make a decsion that ...
We should retain the new Title feature for model training
'''

for dataset in combine:
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# pandas
print ('_'*15 + "train_df crosstab title, sex" + '_'*15)
print (pd.crosstab(train_df['Title'], train_df['Sex']))


for dataset in combine:
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print ('_'*15 + "train_df title, survived groupby" + '_'*15)
print (train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# we can covert the categorical titles to ordinal

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)

print ('_'*15 + "train_df head" + '_'*15)
print (train_df.head())



# NOW, we can safely drop the name feature.
# recall, we also do not need passenger ID

train_df = train_df.drop(["Name", "PassengerId"], axis=1)
test_df = test_df.drop(["Name"], axis=1)
combine = [train_df, test_df]
print ('_'*15 + "train_df shape, test_df shape" + '_'*15)
print (train_df.shape, test_df.shape)

# Converting a categorical Feature
'''
Now we can convert features which contain strings 
to numerical values. This is required by most model algorithms.
Doing so will also help us in achieving the feature
completing goal.
'''

for dataset in combine:
	dataset['Sex'] = dataset['Sex'].map( {"female":1, "male":0} ).astype(int)

print ('_'*15 + "train_df head" + '_'*15)
print (train_df.head())




# completing a numerical continuous feature


guess_ages = np.zeros((2,3))

# now iterate over Sex (0 or 1) and Pclass (1, 2, 3)
# to calculate guessed values of the Age for the 6 combinations

for dataset in combine:
	for i in range(0, 2):
		for j in range(0, 3):
			guess_df = dataset[(dataset['Sex'] == i) &
				(dataset['Pclass'] == j + 1)]['Age'].dropna()

			# age_mean = guess_df.mean()
			# age_std = guess_df.std()
			# age_guess = rnd.uniform(age_mean - age_std,
			# 		age_mean + age_std)

			age_guess = guess_df.median()

			guess_ages[i,j] = int(age_guess / 0.5 + 0.5) * 0.5

	for i in range(0, 2):
		for j in range(0, 3):
			dataset.loc[ (dataset.Age.isnull()) 
			& (dataset.Sex == i) 
			& (dataset.Pclass == j+1),
				'Age'] = guess_ages[i,j]

# pandas . cut the age values into 5 equal chunks of data
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print ('_'*15 + "train_df ageband, survived, groupby" + '_'*15)
print (train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))


for dataset in combine:  
	# dataset . locate [ condition of 'AGE' ] = new value  
	dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

# we categorized the age groups into the age bands,

train_df = train_df.drop(["AgeBand"], axis=1)
combine = [train_df, test_df]
print ('_'*15 + "train_df head" + '_'*15)
print (train_df.head())

# Creating new feature by combining existing features.

for dataset in combine:
	# + 1 for self
	dataset["FamilySize"] = dataset['SibSp'] + dataset['Parch'] + 1



print ('_'*15 + "train_df FamilySize, survived, groupby" + '_'*15)
print (train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# we can create another feature called IsAlone

for dataset in combine:
	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print ('_'*15 + "train_df IsAlone, survived, groupby" + '_'*15)
print (train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]


# we can also create an artificial feature by combining two features
for dataset in combine:
	dataset['Age*Class'] = dataset.Age * dataset.Pclass


print ('_'*15 + "train_df Age*Class" + '_'*15)
print (train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))


# completing a categorical feature
'''
Embarked feature takes S, Q, and C values based on which port.
Our training dataset has two missing values.
We will simply fill these with the most common occurence.
'''

freq_port = train_df.Embarked.dropna().mode()[0]
# most frequent port was S,
for dataset in combine:
	# if dataset at embarked was Null, then fill with freq port
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

print ('_'*15 + "train_df Embarked, survived, groupby" + '_'*15)
print (train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# converting categorical feature to numeric
# we can now convert the embarkeFill feature by creating a new
# numeric Port feature.

for dataset in combine:
	dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


print ('_'*15 + "train_df head" + '_'*15)
print (train_df.head())



# quick completing and coverting a numeric feature

'''
We can no complete the fare feature, by using the mode of the dataset
Note that we are not creating and intermediate new feature.

'''

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

print ('_'*15 + "train_df FareBand, survived, groupby" + '_'*15)
print (train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))


for dataset in combine:
	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
	dataset['Fare'] = dataset['Fare'].astype(int)


train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print ('_'*15 + "train_df head" + '_'*15)
print (train_df.head(10))



'''
- Logistic regression
- KNN
- Support Vector Machines
- Naive Bayes Classifier
- Decision Tree
- Random Forest
- Perceptron
- Artificial Neural Network
- RVM
'''

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

X_test = test_df.drop("PassengerId", axis=1).copy()

'''
Logistic Regression is a useful model to run early in the workflow
LR measures the relationship between the categorical dependent variable (feature)
and one or more independent variables (features) by estimating
probabilities using a logisic function, which is the cumulative
logistice distribution.

Basically,
Positive coefficients increase the log-odds of the response
(and thus increase the probability), 
and negative coefficients decrease the log-odds of the response
(and thus decrease the probability).
'''

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "Logistic Regression Confidence Score" + '_'*15)
print (acc_log)

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print ('_'*15 + "Logistic Regression Feature Coefficients" + '_'*15)
print(coeff_df.sort_values(by="Correlation", ascending=False))



'''
Support Vector Machines are supervised learning models with
associated learning algorithms that analyze data used for classification
and regression analysis.

Given a set of training samples, each marked as belonging to one 
or the other of two categories, and SVM training algorithm builds
a model that assigns new test samples to one category or the
other, making it a non-probabilistic binary linear classifier.

Note that for this case, the SVM generates a higher confidence score
than the LR
'''

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "Support Vector Machines Confidence Score" + '_'*15)
print (acc_svc)

'''
K-Nearest Neighbors algorithm (KNN) is a non-parametric method,
used for classification and regression. A sample is classified by
a majority vote of its neighbors, with the sample being assigned
to the class mos common among its k nearest neighbors.
(k is typically small positive integer)
if k = 1, then the object is simply assigne to the class of that
single nearest neighbor.
'''

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "K-Neighbors Classifier Confidence Score" + '_'*15)
print (acc_knn)

'''
Naive Bayes Classifiers are a family of simple probabilistic Classifiers
based on applying Bayes' theorem with strong(naive) independence
assumptions between the features. Naive Bayes classfiers are highly
scalable, requiring a number of paremeters linear in the number of
variables(features) in a learning problem

In this case, the NBC scores the lowest in the confidence score
'''

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "Naive Bayes Classifiers Confidence Score" + '_'*15)
print (acc_gaussian)

'''
Perceptron is an algorithm fro supervised learning of binary classifiers,
(functions that can decide wheter and input, represented by a Vector
	of numbers, belongs to some specific class or not).
It is a type of linear classifier, i.e. a classification algorithm
that makes its predictions based on linear predictor function 
combining a set of weights with the feature vector. 
The algorithm allows for online learning, in that it processes
elements in the training set one at a time.
'''

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "Perceptron Confidence Score" + '_'*15)
print (acc_perceptron)

'''
Linear SVC
'''

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "Linear SVC Confidence Score" + '_'*15)
print (acc_linear_svc)

'''
Stochastic Gradient Descent
'''

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "Stochastic Gradient Descent Confidence Score" + '_'*15)
print (acc_sgd)

'''
Decision Tree maps features to 'tree branches'
Tree models where the target variable can take a 
finite set of values are called classification trees;
in these tree structures, leaves represent class labels
and branches represent conjunctions of features that
lead to those class labels. 

Decision Trees that take continuous values are called
regression trees

Note decision trees have a habit of overfitting to their
trainer set
'''


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "Decision Tree Confidence Score" + '_'*15)
print (acc_decision_tree)

'''
Random Forests Classifier is most popular. Random forests, or 
random decision forests are an ensemble of learning methods
for classification, regression and other tasks, that operate
by constructing a multitude of decision tress (n_estimators = 100)
at training time, and outputting the class that is the mode of the 
classes (classification) or the mean (regression) of the
individual trees.

Note that the confidence score is the highest among the models
evaluated so far.
'''


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print ('_'*15 + "Random Forest Confidence Score" + '_'*15)
print (acc_random_forest)


'''
Now we rank our evaluations.
'''

models = pd.DataFrame({
	'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
				'Random Forest', 'Naive Bayes', 'Perceptron', 
				'Stochastic Gradient Decent', 'Linear SVC', 
				'Decision Tree'],
	'Score': [acc_svc, acc_knn, acc_log, 
				acc_random_forest, acc_gaussian, acc_perceptron, 
				acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

# Finally,
submission = pd.DataFrame({
		"PassengerId": test_df["PassengerId"],
		"Survived": Y_pred
	})

print ('_'*15 + "Submission" + '_'*15)
print (submission)





