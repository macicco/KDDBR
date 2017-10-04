import numpy as np
import pandas as pd
import sklearn as skl
import imblearn

from sklearn.linear_model import LogisticRegression
from imblearn import over_sampling


import os
import sys

def main(path_outputFile,polinomial_features):
    path_DatasetPerFeature = 'dataset/DatasetPerFeature/'
    featureNames = get_featureNames(path_DatasetPerFeature)
    
    for i in range(len(featureNames)):
        print('Feature set: %s' %(featureNames[i]))
        df_train = pd.read_csv( path_DatasetPerFeature+'train_'+featureNames[i]+'.csv', sep=',') #import the csv file
        df_test = pd.read_csv( path_DatasetPerFeature+'test_'+featureNames[i]+'.csv' ,sep=',') #import the csv file

        probas_testKaggle = executeClassifier(df_train, df_test, polinomial_features) #get the probabilities of the test examples 
        probas_trainKaggle = executeClassifier_leaveOneOut(df_train, polinomial_features) #get the probabilities of the train examples       
       
        if i==0:
            #inicialize pandas dataframes to save the train probs and test probs            
            df_trainProbs = pd.DataFrame(columns=['Id']+featureNames+['Class'])
            df_testProbs = pd.DataFrame(columns=['Id']+featureNames)
            
            df_trainProbs['Class'] = df_train['Class'].values
            
            df_trainProbs['Id'] = df_train.iloc[:,0].values
            df_testProbs['Id'] = df_test.iloc[:,0].values
            
        df_trainProbs[featureNames[i]] = probas_trainKaggle[:,1]
        df_testProbs[featureNames[i]] = probas_testKaggle[:,1]
    
    probas_testKaggle = executeClassifier(df_trainProbs, df_testProbs, polinomial_features) #get the probabilities of the test examples based on the new feature set 
    
    print_output_KaggleFormat(probas_testKaggle, df_testProbs, path_outputFile)

def get_featureNames(path_DatasetPerFeature):
    
    files = os.listdir(path_DatasetPerFeature)
    featuresNames = []; 
    for file in files:
        if 'train' in file:
            file = file.replace('train_','')
            file = file.replace('.csv','')
            
            featuresNames.append(file)
    
    featuresNames.sort()
    
    return featuresNames
 
def executeClassifier(df_train, df_test, polinomial_features):
    
    x_train = df_train.iloc[:,1:-1].values   
    x_train = np.nan_to_num(x_train)
    y_train = df_train[ df_train.columns[-1] ].values  
                 
    x_test_kaggle = df_test.iloc[:,1:].values 
    x_test_kaggle = np.nan_to_num(x_test_kaggle)                                                                 
    
    x_train2 = x_train[:] #get a copy of x_train
    probas_testKaggle = train_predict( x_train2, y_train, x_test_kaggle,  polinomial_features) #it performs training and classification
        
    return probas_testKaggle 

def executeClassifier_leaveOneOut(df_train, polinomial_features):
   
    data = df_train.iloc[:,1:-1].values   
    data = np.nan_to_num(data)
    target = df_train[ df_train.columns[-1] ].values  
       
    cv = skl.model_selection.LeaveOneOut()
    
    i = 0
    y_test = np.zeros( len(target),  dtype=int )
    probas = np.zeros( (len(target),2) )
    for train_index, test_index in cv.split(data):     
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test[i] = target[train_index], target[test_index]
        
        probas[i,:] = train_predict( x_train, y_train, x_test, polinomial_features) #it performs training and classification
        
        i+=1
    
    return probas
       
def train_predict( x_train, y_train, x_test, polinomial_features):
    
    x_train = x_train.astype(float) #convert the values to float
    x_test = x_test.astype(float) #convert the values to float
    
    classifier = skl.linear_model.LogisticRegression(random_state = 5)  

    ########################################################################
    if polinomial_features:
        #print('polinomial')
        poly = skl.preprocessing.PolynomialFeatures(degree=2, interaction_only=False)
        x_train = poly.fit_transform(x_train)
        x_test = poly.transform(x_test)
    ########################################################################

    #standardization of the datasets
    scaler = skl.preprocessing.StandardScaler().fit(x_train) # fit to the train data
    x_train = scaler.transform(x_train) # standardization of the train data
    x_test = scaler.transform(x_test) # standardization of the test data
    
    #class balancing using the SMOTE method
    sm = imblearn.over_sampling.SMOTE(random_state=10, ratio = 'minority') 
    x_train, y_train = sm.fit_sample(x_train, y_train) #create synthetic minority class examples to balance the training set
    
    classifier.fit(x_train, y_train) #train the classifier
    probas_ = classifier.predict_proba(x_test) #classify the test examples
  
    return probas_

def print_output_KaggleFormat(probs_testKaggle, df_test_kaggle, pathResults):   

    fileNew = open(pathResults,'w')#abre para gravar 
    
    fileNew.write("Id,Prob1,Prob2\n") 
    for i in range(len(probs_testKaggle)):
        fileNew.write("%d,%1.2f,%1.2f\n" %( df_test_kaggle.iloc[i,0],probs_testKaggle[i,0],probs_testKaggle[i,1]))        

    fileNew.close()
     
def showHelp():
    print('\n====================================================')
    print('Usage: python main.py [options] [outputFile]')
    print('\noutputFile:')
    print('\tpath to the output file')
    print('\nOptions:')
    print('\n-p polynomial_features: (default 0)')
    print('    0 - false (do not generate polynomial features)')
    print('    1 - true (generate polynomial features)')    
    print('====================================================\n')
    
  
    
       
if __name__ == "__main__":
    
    polinomial_features = False #(default parameter)
    
    if len(sys.argv)<2 or len(sys.argv)>4:
        showHelp()
    elif len(sys.argv)==4:
        if (sys.argv[1] == '-p') and (sys.argv[2] == '1' or sys.argv[2] == '0'):
            if sys.argv[2] == '1': 
                polinomial_features = True
                
            outputFile = sys.argv[3]
            main(outputFile,polinomial_features)
        else: 
            showHelp()
    elif len(sys.argv)==2:
        outputFile = sys.argv[1]
        main(outputFile,polinomial_features)
    else: 
        showHelp()    
            
