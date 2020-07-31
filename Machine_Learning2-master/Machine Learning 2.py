import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import numpy as np
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('ML2_Data.csv')
data1 = data.describe()
#print(data1, "\n\n\n")

#Converts normal and abnormal to True andFalse
replacement = {'Abnormal': False, 'Normal': True}
data['Status'] = data['Status'].map(replacement)

#############################################################################################################
print("There are" , len(data.index), "data values in each feature")
print("There are" , len(data.columns), "features")
print("Are there any Null data:",data.isnull().values.any())
print("There is only 1 catagorical variable which contains 2 possible values of Normal and Abnormal\n\n")

#Creates 2 dataframes containing the normal and abnormal data
#for the Vibration_sensor_1 Feature
trueVS1 = data[data['Status']==True].loc[:, ['Vibration_sensor_1']]
falseVS1 = data[data['Status']==False].loc[:, ['Vibration_sensor_1']]

#Creates 2 subplots and boxplots the data
fig, (TrueValues, FalseValues) = plt.subplots(1,2)
fig.suptitle('Vibration_sensor_1 data for Normal and Abnormal')
TrueValues.boxplot(trueVS1['Vibration_sensor_1'])
FalseValues.boxplot(falseVS1['Vibration_sensor_1'])

#Sets up density plot 
tempFrame = pd.DataFrame({'Normal': trueVS1['Vibration_sensor_1'], 'Abnormal': falseVS1['Vibration_sensor_1']})

tempFrame.plot.kde()

#############################################################################################################

print("DONE!")



#=============================NEURAL NETWORK============================
from keras.models import Sequential
from keras.layers import Dense

#Import data
data = pd.read_csv('ML2_Data.csv')
#Shuffles Data
data = data.sample(frac=1).reset_index(drop = True)

#Replaces normal and abnormal with boolean
replacement = {'Abnormal': False, 'Normal': True}
data['Status'] = data['Status'].map(replacement)

#Normalises data
# x = data.values
# x_scaled = preprocessing.MinMaxScaler().fit_transform(x)
# data = pd.DataFrame(x_scaled)

#Calculates Test Train Split
testTrainSplit = len(data.index) * 0.9

#Splits data into test and train
trainData = data.iloc[:int(testTrainSplit)]
testData = data.iloc[int(testTrainSplit):]

#Creates train X and Y data
trainDataX = trainData.loc[:, trainData.columns != 'Status'].as_matrix()
trainDataY = trainData.loc[:, ['Status']].as_matrix()

#Creates test X and Y data
testDataX = testData.loc[:, testData.columns != 'Status'].as_matrix()
testDataY = testData.loc[:, ['Status']].as_matrix()






# ===============================NEURAL NET ========================================

def NeuralNetwork(trainX, trainY, testX, testY, numEpochs, numNodes, batch, test):
    NN = Sequential()
    NN.add(Dense(int(numNodes), input_dim = 12, activation='sigmoid'))
    NN.add(Dense(int(numNodes),  activation='sigmoid'))
    NN.add(Dense(1,  activation='sigmoid'))

    NN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    NN.fit(trainX, trainY, epochs=int(numEpochs), batch_size = int(batch), verbose = 0)

    if test == True:
        accuracy = NeuralNetworkTest(NN, testX, testY)
        return accuracy
    else:
        return NN

def NeuralNetworkTest(NeuralNet,testX, testY):
    _, accuracy = NeuralNet.evaluate(testX, testY)
    print('NN Test Accuracy: %.2f' % (accuracy*100))
    
    return accuracy

epochs = ["1", "5", "10", "50", "100", "500", "1000"]
accuracies = []

for i in range(len(epochs)):   
    accuracies.append(NeuralNetwork(trainDataX, trainDataY, testDataX, testDataY, int(epochs[i]), 1000, 30, True))

plt.plot(epochs, accuracies, 'r')
plt.title("Accuracys for increasing Epochs")
plt.show()


print("DONE!")



#=================================Trees==================================
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def Trees(xDataTrain, yDataTrain, xDataTest, yDataTest, numTrees, numLeaf):
    rndTrees = RandomForestClassifier(n_estimators = int(numTrees), min_samples_leaf = numLeaf)
    rndTrees.fit(xDataTrain, yDataTrain)
    accuracy = metrics.accuracy_score(yDataTest, rndTrees.predict(xDataTest)) * 100
    print("Tree Accuracy: ", accuracy, "%")
    
    return accuracy

trees = ["10", "50", "100", "500", "1000", "5000", "10000"]
accuracy5 = []
accuracy50 = []

for i in range (len(trees)):
    accuracy5.append(Trees(trainDataX, trainDataY, testDataX, testDataY, trees[i], 5))
    accuracy50.append(Trees(trainDataX, trainDataY, testDataX, testDataY, trees[i], 50))
    
plt.plot(trees, accuracy5, 'r', label = '5 Leaves')
plt.plot(trees, accuracy50, 'b', label = '50 Leaves')
plt.title('Number of Trees againt Accuracy')
plt.legend()
plt.show()



from sklearn.model_selection import StratifiedKFold
#======================K-Fold=====================================
kFold = StratifiedKFold(n_splits = 10, shuffle = False)

dataX = data.loc[:, data.columns != 'Status'].as_matrix()
dataY = data.loc[:, ['Status']].as_matrix()

#=====================Neural Network and Trees=====================

numNodes = ["50", "500", "1000"]
numTrees = ["20", "500", "10000"]

cVScoresNN = []
cVScoresTrees  =[]

for i in range(len(numNodes)):  
    
    print("\n\nThis is for numNodes: ", numNodes[i])
    print("This is for numTrees: ", numTrees[i])
    accuracyNN = []
    accuracyTree = []
    
    for train,test in kFold.split(dataX, dataY):        
        accuracyNN.append(NeuralNetwork(dataX[train], dataY[train], dataX[test], dataY[test], 500, numNodes[i], len(dataX[train]), True))
        print("NN DONE")
        accuracyTree.append(Trees(dataX[train], dataY[train], dataX[test], dataY[test], numTrees[i], 5))
        print("TREE DONE")

    totalAccuracyNN = np.mean(accuracyNN) * 100
    totalAccuracyTrees = np.mean(accuracyTree)
    
    cVScoresNN.append(totalAccuracyNN)
    cVScoresTrees.append(totalAccuracyTrees)

print("\n\n")

print("NN Accuracy 50: " , cVScoresNN[0])
print("NN Accuracy 500: " , cVScoresNN[1])
print("NN Accuracy 1000: " , cVScoresNN[2])

plt.plot(numNodes, cVScoresNN, 'b')
plt.show()

print("\n\n")
    
print("Tree Accuracy 20: " , cVScoresTrees[0])
print("Tree Accuracy 500: " , cVScoresTrees[1])
print("Tree Accuracy 10000: " , cVScoresTrees[2])

plt.plot(numTrees, cVScoresTrees, 'b')
plt.show()

print("DONE!")