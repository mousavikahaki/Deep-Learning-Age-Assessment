import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.decomposition import PCA
import keras
import common_functions as cf
############################Read Data############################

DentalFeatures = pd.read_csv("DentalFeatures.csv",sep=",")
DentalC2Target = pd.read_csv("Target.csv",sep=",")

############################Generate Input, Labels, and Train and Test Data############################

# Specify the data only (without Labels)
X=DentalFeatures
# PCA
# pca = PCA(n_components=4)
# X = pca.fit_transform(X)

# Specify the Target labels and flatten the array (Labels)
y=DentalC2Target
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

############################Normalize Data############################

# Define the scaler
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

############################Create the model############################

# Initialize the constructor
model = Sequential()  # comes from import: from keras.models import Sequential
# Add an input layer
model.add(Dense(X.shape[1], activation='relu', input_shape=(X.shape[1],)))
# Add one hidden layer
model.add(Dense(8, activation='relu'))
# Add an output layer
model.add(Dense(y.shape[1], activation='sigmoid'))
# Model output shape
model.output_shape
# Model summary
model.summary()
# Model config
model.get_config()
# List all weight tensors
model.get_weights()

# class SimpleMLP(keras.Model):
#
#     def __init__(self, use_bn=False, use_dp=False, num_classes=2):
#         super(SimpleMLP, self).__init__(name='mlp')
#         self.use_bn = use_bn
#         self.use_dp = use_dp
#         self.num_classes = num_classes
#
#         self.dense1 = keras.layers.Dense(79, activation='relu')
#         self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
#         if self.use_dp:
#             self.dp = keras.layers.Dropout(0.5)
#         if self.use_bn:
#             self.bn = keras.layers.BatchNormalization(axis=-1)
#
#     def call(self, inputs):
#         x = self.dense1(inputs)
#         if self.use_dp:
#             x = self.dp(x)
#         if self.use_bn:
#             x = self.bn(x)
#         return self.dense2(x)
#
# model = SimpleMLP()
# # model.compile(...)
# # model.fit(...)

############################Compile and fit the Model############################

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

############################Save Model############################

# # detect the current working directory and print it
# path = cf.os.getcwd()
# path = path+'\model'
# access_rights = 777
#
# cf.createFolder('/Training_result/')
#
# model.save(path)
#
# ############################Load Model############################
# model1 = load_model('/model')


############################Predict Values############################
y_pred = model.predict(X_test)

############################Evaluate Model############################

score = model.evaluate(X_test, y_test,verbose=1)
print(score)

############################Confusion matrix############################

if y.shape[1] ==1:
    conf = confusion_matrix(y_test.round(), y_pred.round())
else:
    conf = confusion_matrix(y_test.round().values.argmax(axis=1), y_pred.round().argmax(axis=1))
print("Confusion Matrix: ",conf)

############################Precision############################

if y.shape[1] ==1:
    precision = precision_score(y_test.round(), y_pred.round())
else:
    precision = precision_score(y_test.round().values.argmax(axis=1), y_pred.round().argmax(axis=1),average='micro') #  average=Nonefor precision from each class
print("Precision: ",precision)

############################Recall############################

if y.shape[1] ==1:
    recall= recall_score(y_test.round(), y_pred.round())
else:
    recall = recall_score(y_test.round().values.argmax(axis=1), y_pred.round().argmax(axis=1),average='micro')
print("Recall: ",recall)

############################F1 score############################

if y.shape[1] ==1:
    f1_score = f1_score(y_test.round(),y_pred.round())
else:
    f1_score = f1_score(y_test.round().values.argmax(axis=1), y_pred.round().argmax(axis=1),average='micro')
print("F1 Score: ",f1_score)

############################Cohen's kappa############################

if y.shape[1] ==1:
    cohen_kappa_score = cohen_kappa_score(y_test.round(), y_pred.round())
else:
    cohen_kappa_score = cohen_kappa_score(y_test.round().values.argmax(axis=1), y_pred.round().argmax(axis=1))
print("Cohen_Kappa Score: ",cohen_kappa_score)

############################End############################

print("Done!")