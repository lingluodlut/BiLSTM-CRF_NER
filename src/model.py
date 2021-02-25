# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:53:40 2021

@author: luol2
"""


from representation import RepresentationLayer
from keras.layers import *
from keras.models import Model
from keras_contrib.layers import CRF


class BiLSTM_CRF():
    def __init__(self, model_files):

        self.fea_dict = {'word': 1,
                         'char': 1}
    
        self.hyper = {'sen_max'      :200,
                      'word_max'     :40,
                      'charvec_size' :50}
             
        self.w2vfile=model_files['w2vfile']      
        self.charfile=model_files['charfile']
        self.labelfile=model_files['labelfile']
          
        vocab={'char':self.charfile,'label':self.labelfile}
        print('loading w2v model.....') 
        self.rep = RepresentationLayer(self.w2vfile,vocab_file=vocab, frequency=400000)
         
        print('building  model......')
        all_fea = []
        fea_list = []
        
        if self.fea_dict['word'] == 1:
            word_input = Input(shape=(self.hyper['sen_max'],), dtype='int32', name='word_input')  
            all_fea.append(word_input)
            word_fea = Embedding(self.rep.vec_table.shape[0], self.rep.vec_table.shape[1], weights=[self.rep.vec_table], trainable=True,mask_zero=False, input_length=self.hyper['sen_max'], name='word_emd')(word_input)
            fea_list.append(word_fea)
    
        if self.fea_dict['char'] == 1:
            char_input = Input(shape=(self.hyper['sen_max'],self.hyper['word_max']), dtype='int32', name='char_input')
            all_fea.append(char_input)
            char_fea = TimeDistributed(Embedding(self.rep.char_table_size, self.hyper['charvec_size'], trainable=True,mask_zero=False),  name='char_emd')(char_input)
            char_fea = TimeDistributed(Conv1D(self.hyper['charvec_size']*2, 3, padding='same',activation='relu'), name="char_cnn")(char_fea)
            char_fea_max = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling_max")(char_fea)
            fea_list.append(char_fea_max)
              
    
        if len(fea_list) == 1:
            concate_vec = fea_list[0]
        else:
            concate_vec = Concatenate()(fea_list)
    
        concate_vec = Dropout(0.4)(concate_vec)
    
        # model
        bilstm = Bidirectional(LSTM(200, return_sequences=True, implementation=2, dropout=0.4, recurrent_dropout=0.4), name='bilstm1')(concate_vec)
        dense = TimeDistributed(Dense(200, activation='tanh'), name='dense1')(bilstm)
    
        dense= Dropout(0.4)(dense)
        crf = CRF(self.rep.label_table_size, sparse_target=True)
        output = crf(dense)
        self.model = Model(inputs=all_fea, outputs=output)
        self.crf= crf
        
    def load_model(self,model_file):
        self.model.load_weights(model_file)
        self.model.summary()        
        print('load model done!')