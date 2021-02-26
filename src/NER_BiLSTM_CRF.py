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
    
    train_x, train_y = nn_model.rep.represent_instances_fea(train,train_label,word_max_len=nn_model.hyper['sen_max'],char_max_len=nn_model.hyper['word_max'])
    test_x, test_y = nn_model.rep.represent_instances_fea(test,test_label,word_max_len=nn_model.hyper['sen_max'],char_max_len=nn_model.hyper['word_max'])

    input_train=[]
    input_test=[]

    if nn_model.fea_dict['word'] == 1:
        input_train.append(train_x[0])
        input_test.append(test_x[0])

    if nn_model.fea_dict['char'] == 1:
        input_train.append(train_x[1])
        input_test.append(test_x[1])

    #print(input_train[0][0],input_train[1][0])
    #print(train_y[0])

    # train the model
    opt = Adam(lr=0.005) 
    nn_model.model.compile(loss=nn_model.crf.loss_function, optimizer=opt,metrics=[nn_model.crf.accuracy])
    nn_model.model.summary()
    nn_model.model.fit(input_train,train_y,batch_size=256, epochs=30,verbose=1)

    # test the model
    test_predict = nn_model.model.predict(input_test)
    out_BIO(outfiles['test_out'],test_predict,test,nn_model.rep.index_2_label)
    commond='./conlleval.pl < '+outfiles['test_out']
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
        

