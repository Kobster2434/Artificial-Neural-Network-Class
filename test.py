from ANN import ANN, Dense
import pandas as pd
from sklearn.utils import resample

# Upscaling the data.
dataset = pd.read_csv('Churn_Modelling.csv')
majority = dataset[dataset.Exited==0]
minority = dataset[dataset.Exited==1]
minority_upsampled = resample(minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=7963,    # to match majority class
                                 random_state=123) # reproducible results
dataset2 = pd.concat([majority, minority_upsampled])

X = dataset2.iloc[:, 3:13].values
y = dataset2.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) # One hot encoder to create dummy variables
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # To avoid the dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler # Compulsary in neural networks (no independent variables dominating eachother)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = ANN()

#classifier.add(Dense(input_shape = 11, units = 12, activation = "sigmoid", 
#                     init_weights = "random_s", init_bias = "random_s"))
#classifier.add(Dense(input_shape = 12, units = 13, activation = "sigmoid", 
#                     init_weights = "random_s", init_bias = "random_s"))
#classifier.add(Dense(input_shape = 13, units = 2, activation = "softmax",
#                     init_weights = "random_s", init_bias = "random_s"))
#classifier.compile("mse", 0.01)


classifier.fit(X_train, y_train, batch_size = 300, epochs = 100)

_, accuracy = classifier.evaluation(X_test, y_test)
print(accuracy)

_, accuracy_train = classifier.evaluation(X_train, y_train)
print(accuracy_train)

counter = 0
for i in y:
    if y[i] == 1:
        counter += 1
print(counter/X.shape[0])

'''
y_pred = classifier.predict(X_train)
pred = []
for i in range(len(y_pred)):
    if y_pred[i] > 0.5:
        pred.append(1)
    else: 
        pred.append(0)
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, pred)
'''




# classifier.layers[-1].weights
# array([[0.01387328, 0.06279189, 0.06851785, 0.04335786, 0.0170582 ,
       # 0.03010248]])
