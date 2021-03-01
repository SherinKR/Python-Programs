# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:13:23 2020

@author: Sherin KR 
"""
###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn, tensorflow
import xlsxwriter 
import xlrd
from sklearn import svm #, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import model_selection
#from keras.layers import containers
 

#import tensorflow as tf
#tf.python.control_flow_ops = tf
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io
#from keras.layers.core import  AutoEncoder
from keras.layers import merge

from keras.layers import Input, Dense
from keras.engine.training import Model
from keras.models import Sequential, model_from_config
from keras.layers.core import  Dropout, Activation, Flatten#, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.constraints import maxnorm
#from deepfun import multiple_layer_autoencoder,autoencoder_two_subnetwork_fine_tuning


def prepare_data(seperate=False):
    print ("loading data")
    
    
    miRNA_fea = np.loadtxt("DrugSimMat_50.txt",dtype=float,delimiter=",")
    disease_fea = np.loadtxt("DiseaseSimMat_50.txt",dtype=float,delimiter=",")
   
    interaction = np.loadtxt("DiDrAMat_50.txt",dtype=int,delimiter=",")
    
    '''
    
    miRNA_fea = np.loadtxt("Integrated_miR_sim 540_50.txt")
    disease_fea = np.loadtxt("disease similarity matrix_50.txt")
   
    interaction = np.loadtxt("Drug-MiRNA association matrix_50.txt")#,delimiter=None)

    miRNA_fea = np.loadtxt("drug similarity matrix 831.txt")  
    disease_fea = np.loadtxt("Integrated_miR_sim 540.txt")
    #interaction = np.loadtxt("miRNA_disease_matrix2.txt",dtype=int,delimiter=" ")
    interaction = np.loadtxt("Drug-MiRNA association matrix.txt")
    #disease_fea2 = np.loadtxt("disease_sim_integrate_chose_one.txt",dtype=float,delimiter=",")
   
     ''' 
    
  
    
    
    link_number = 0
    train = []         
    label = []
    link_position = []
    nonLinksPosition = []  # all non-link positions
    
    for i in range(0, interaction.shape[0]):   # shape[0] returns m if interaction is m*n, ie, returns no. of rows of matrix
        for j in range(0, interaction.shape[0]):  # returns no. of columns of matrix interaction, here 495, interaction = 50*495
            label.append(interaction[i,j])    #append an item at the end of the list,ie, add the labels 1 or 0, add all the elements of interaction matrix one by one to list label
            if interaction[i, j] == 1:
                link_number = link_number + 1     #counts no of existing associations
                link_position.append([i, j])      #stores matrix indices of existing associations
                miRNA_fea_tmp = list(miRNA_fea[i])  # miRNA_fea_temp gets a vector corresp to each miRNAs.
                disease_fea_tmp = list(disease_fea[j])	#list() = converts a sequence to list. stores each row of similarity matrix as lists
				    # disease_fea_tmp = list(disease_fea[j])+list(disease_fea2[j]) hide
                
            elif interaction[i,j] == 0:
                nonLinksPosition.append([i, j])   #stores indices corresponding to 0's in interaction matrix
                miRNA_fea_tmp = list(miRNA_fea[i])
		               
                disease_fea_tmp = list(disease_fea[j])
				        #disease_fea_tmp = list(disease_fea[j])+list(disease_fea2[j]) hide
            if seperate:
                tmp_fea = (miRNA_fea_tmp,disease_fea_tmp)
            else:
                tmp_fea = miRNA_fea_tmp + disease_fea_tmp    #tmp_fea is a list(tuple)
            train.append(tmp_fea)     #train contains all the miRNA features, then contains all the disease features
    return np.array(train), label   # train became an array,label contains all the elements of association matrix as a sequence

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        sensitivity =  float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
    else:
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
       
    return acc, precision, sensitivity, specificity, MCC 

def transfer_array_format(data):    #data=X  , X= all the miRNA features, disease features 
    formated_matrix1 = []
    formated_matrix2 = []
    #pdb.set_trace()
    #pdb.set_trace()
    for val in data:
        #formated_matrix1.append(np.array([val[0]]))
        formated_matrix1.append(val[0])   #contains miRNA features ?
        formated_matrix2.append(val[1])   #contains disease features ?
        #formated_matrix1[0] = np.array([val[0]])
        #formated_matrix2.append(np.array([val[1]]))
        #formated_matrix2[0] = val[1]      
    
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def DNN():
    model = Sequential()
    model.add(Dense(input_dim=906, output_dim=500))#, 1371,878 shapeinit='glorot_normal')) ## 1027 1261 1021 918 128 878 638 535
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    #model.add(Dense(input_dim=300, output_dim=300,init='glorot_normal'))  ##500
    #model.add(Activation('relu'))
    #model.add(Dropout(0.3))

    model.add(Dense(input_dim=500, output_dim=300))#,init='glorot_normal'))  ##500
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(input_dim=300, output_dim=2))#,init='glorot_normal'))  ##500
    model.add(Activation('softmax'))
    #sgd = SGD(l2=0.0,lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta) #, class_mode="binary")##rmsprop sgd
    return model





def DeepMDA():
    X, labels = prepare_data(seperate = True)     #X= array of concatinated features,labels=corresponding labels?
    
    #import pdb            #python debugger
    
    X_data1, X_data2 = transfer_array_format(X) # X-data1 = miRNA features(2500*495),  X_data2 = disease features (2500*383)
    #print (X_data1.shape,X_data2.shape)
    y, encoder = preprocess_labels(labels)# labels labels_new
    num = np.arange(len(y))
    np.random.shuffle(num)
    X_data1 = X_data1[num]
    X_data2 = X_data2[num]
    y = y[num]
    
    num_cross_val = 2
    all_performance = []
    all_performance_rf = []
    all_performance_bef = []
    all_performance_DNN = []
    all_performance_SDADNN = []
    all_performance_blend = []
    all_labels = []
    all_prob = {}
    num_classifier = 3
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_averrage = []
    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        print("$$$$$$$$$$$$",test1)
        print(test2)
          
        real_labels = []
        for val in test_label:
            if val[0] == 1:             #tuples in array, val[0]- first element of tuple
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        class_index = 0
        
       

        ## DNN 
        class_index = class_index + 1
        prefilter_train = np.concatenate((train1, train2), axis = 1)
        prefilter_test = np.concatenate((test1, test2), axis = 1)
        
        model_DNN = DNN()
        train_label_new_forDNN = np.array([[0,1] if i == 1 else [1,0] for i in train_label_new])
        print (train_label_new_forDNN[0])
        
        model_DNN.fit(prefilter_train,train_label_new_forDNN,batch_size=200,nb_epoch=20,shuffle=True) #####give input here
        proba = model_DNN.predict_classes(prefilter_test,batch_size=200,verbose=True)
        proba2 = model_DNN.predict(prefilter_test,batch_size=200,verbose=True)
        
       
        
        print("*********")
        print(proba2)
        print("*********")
        
        
        
        
         #**********************************
        #Writing result to excel sheet named RESULT
        #********************************
        
        interaction = np.loadtxt("DiDrAMat_50.txt",dtype=int,delimiter=",")
        
        drug = xlrd.open_workbook(r"D:\Project\Main Project\Final\Drug names.xlsx")
                
        disease= xlrd.open_workbook(r"D:\Project\Main Project\Final\disease names.xlsx")
        
        sheet1 = drug.sheet_by_index(0) 
        #XX= sheet1.cell_value(0, 1) 
        #print(XX)
        sheet2 = disease.sheet_by_index(0)        
                      
        
        
        workbook = xlsxwriter.Workbook('RESULT.xlsx') 
  
        worksheet = workbook.add_worksheet()                              
                          
            
           
        p= 0
        rows = 0
        columns =0
        cl = 1
        
        for i in range(0, interaction.shape[0]):  
              for j in range(0, interaction.shape[0]):
                   if interaction[i,j]== 0:
                      x= sheet1.cell_value(i, 1)    # writing drug names in the excel first column
                      worksheet.write(rows, columns, x)
                      
                      y= sheet2.cell_value(j, 1)  # Writing disease names in the excel second column;Thease 2 lines got problem, correct it. 
                      worksheet.write(rows, cl, y)
                      
                      rows += 1
                      p = p+1
                      
        
                      
        rows = 0
        columns =2   # writing probabilities in the wxcel 3rd column
        cl = 0
        
        for item in proba2 :
            
             worksheet.write(rows, columns, str(item)) 
             rows += 1
       
        workbook.close() 
  
         #**************Finished writing to excel       
      
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test,batch_size=200,verbose=True)
        
       
        
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
      
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob[:,1])
        auc_score = auc(fpr, tpr)
        scipy.io.savemat('raw_DNN',{'fpr':fpr,'tpr':tpr,'auc_score':auc_score})


        ## AUPR score add 
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob[:,1])
        aupr_score = auc(recall, precision1)
        print ("RAW DNN:",acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score)
        all_performance_DNN.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
       
    print ('mean performance of DNN using raw feature')
    print (np.mean(np.array(all_performance_DNN), axis=0))
    print ('---' * 20)
    print("*******ACCURACY*********=",acc)
   
    #print (X_data1.shape,X_data2.shape)  #X_data1= 2500*495, X_data2=2500*383
    #print(X.shape)  # X = 2500*2
   
    
   

def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label


if __name__=="__main__":
    DeepMDA()
    
