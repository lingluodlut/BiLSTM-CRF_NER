# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:49:09 2021

@author: luol2
"""

import sys
import subprocess
from model import BiLSTM_CRF
from processing_data import ml_intext,out_BIO

from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad

def NN_training(vocabfiles,infiles,outfiles):
    

    
    nn_model=BiLSTM_CRF(vocabfiles)

    #load dataset
    print('loading dataset......')
    
    trainfile=infiles['trainfile']
    testfile=infiles['testfile']
    train,train_label = ml_intext(trainfile)
    test,test_label = ml_intext(testfile)
    
    #print(train[0:2])
    #print(train_label[0:2])

    train_over=0
    test_over=0
    for s in train:
        if len(s)>200:
            train_over+=1
    for s in test:
        if len(s)>200:
            test_over+=1  
    train_maxlen = max(len(s) for s in train)
    test_maxlen = max(len(s) for s in test)
    print('train_maxlen:',train_maxlen,'test_maxlee:',test_maxlen,'>200:',train_over,test_over)
    

    
    print('numpy dataset......')
    train_x, train_y = nn_model.rep.represent_instances_fea(train,train_label,word_max_len=nn_model.hyper['sen_max'],char_max_len=nn_model.hyper['word_max'])
    test_x, test_y = nn_model.rep.represent_instances_fea(test,test_label,word_max_len=nn_model.hyper['sen_max'],char_max_len=nn_model.hyper['word_max'])

    #print(train_x[0:2])
    #print(train_y[0:2])

    # train_x, train_y = cnn_model.rep.represent_instances_all_feas(train_set,train_label,word_max_len=cnn_model.hyper['sen_max'],char_max_len=cnn_model.hyper['word_max'])
    # input_train = []
    input_train=[]
    input_test=[]

    if nn_model.fea_dict['word'] == 1:
        input_train.append(train_x[0])
        input_test.append(test_x[0])

    if nn_model.fea_dict['char'] == 1:
        input_train.append(train_x[1])
        input_test.append(test_x[1])




    # opt = Adadelta()
    opt = Adam(lr=0.005) 
    #opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
    nn_model.model.compile(loss=nn_model.crf.loss_function, optimizer=opt,metrics=[nn_model.crf.loss_function])
    nn_model.model.summary()

    
    max_dev=0.0
    max_dev_epoch=0
    max_test=0.0
    max_test_epoch=0

    for i in range(50):
        print('\nepoch:',i)
        nn_model.model.fit(input_train,train_y,batch_size=256, epochs=1,verbose=2)
        if i%2==0:
            test_predict = nn_model.model.predict(input_test,batch_size=50)
            out_BIO(outfiles['test_out']+str(i),test_predict,test,nn_model.rep.index_2_label)
            commond='./conlleval.pl < '+outfiles['test_out']+str(i)
            retval = subprocess.call(commond, shell=True)
    nn_model.model.save(outfiles['model_out'])


            

if __name__=="__main__":


    vocabfiles={'w2vfile':'//panfs/pan1/bionlp/lulab/luoling/bio_nlp_vec/hpo_w2v/bio_embedding_intrinsic.d200',   
                'charfile':'../vocab/char.vocab',
                'labelfile':'../vocab/cdr_label.vocab'}
    
    infiles={'trainfile':'../data/cdr_train_dev_BIO_allfea.tsv',
             'testfile':'../data/cdr_test_BIO_allfea.tsv'}
    
    outfiles={'test_out':'../output/cdr_prediction_BIO.tsv',
              'model_out':'../output/BiLSTM_CRF.h5'}
    NN_training(vocabfiles,infiles,outfiles)
        

