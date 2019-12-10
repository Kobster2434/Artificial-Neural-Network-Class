from ANN import ANN, Dense
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler # Compulsary in neural networks (no independent variables dominating eachother)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = ANN()

classifier.add(Dense(input_shape = 11, units = 10, activation = "sigmoid"))
classifier.add(Dense(input_shape = 10, units = 2, activation = "softmax"))

classifier.compile("mse", 0.01)

classifier.fit(X_train, y_train, batch_size = 100, epochs = 24)

classifier.evaluation(X_test, y_test)