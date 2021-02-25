# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:59:18 2021

@author: luol2
"""


import os, sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences

class RepresentationLayer(object):
    
    
    def __init__(self, wordvec_file,  vocab_file=[],\
                 vec_size=50, word_size=10000, frequency=10000):
        
        '''
        wordvec_file    ：    the file path of word embedding
        vec_size        :    the dimension size of word vector 
                             learned by word2vec tool
        
        word_size       :    the size of word vocabulary
  
        frequency       :    the threshold for the words left according to
                             their frequency appeared in the text
                             for example, when frequency is 10000, the most
                             frequent appeared 10000 words are considered
        

        
        '''
        #load word embedding
        file = open(wordvec_file)
        first_line = file.readline().strip()
        file.close()
        self.word_size = int(first_line.split()[0])
        self.vec_size = int(first_line.split()[1])
        self.frequency = frequency
        
        if self.frequency>self.word_size:
            self.vec_table = np.zeros((self.word_size + 2, self.vec_size))
        else:
            self.vec_table = np.zeros((self.frequency + 2, self.vec_size))
        self.word_2_index = {}
        self.load_wordvecs(wordvec_file)
        
        #other fea
        self.char_2_index={}
        self.char_table_size=0
        if 'char' in vocab_file.keys():
            self.load_fea_vocab(vocab_file['char'],self.char_2_index)
            self.char_table_size=len(self.char_2_index)
            print(self.char_table_size)
            #print(self.char_2_index) 
            
            
        self.label_2_index={}
        self.index_2_label={}
        self.label_table_size=0
        if 'label' in vocab_file.keys():
            self.load_label_vocab(vocab_file['label'],self.label_2_index,self.index_2_label)
            self.label_table_size=len(self.label_2_index)
            print(self.label_table_size)
    
    def load_wordvecs(self, wordvec_file):
        
        file = open(wordvec_file,'r',encoding='utf-8')
        file.readline()
        print(self.word_size)
        print(self.vec_size)
        row = 0
        self.word_2_index['padding_0'] = row #oov-zero vector
        row+=1
        for line in file:
            if row <= self.word_size and row <= self.frequency:
                line_split = line.strip().split(' ')
                self.word_2_index[line_split[0]] = row
                for col in range(self.vec_size):
                    self.vec_table[row][col] = float(line_split[col + 1])
                row += 1
            else:
                break
        
        self.word_2_index['sparse_vectors'] = row #oov-zero vector

        
        file.close()

    def load_fea_vocab(self,fea_file,fea_index):
        fin=open(fea_file,'r',encoding='utf-8')
        i=0
        fea_index['padding_0']=i
        i+=1
        fea_index['oov_padding']=i
        i+=1
        for line in fin:
            fea_index[line.strip()]=i
            i+=1
        fin.close()

    def load_label_vocab(self,fea_file,fea_index,index_2_label):
        fin=open(fea_file,'r',encoding='utf-8')
        all_text=fin.read().strip().split('\n')
        fin.close()
        for i in range(0,len(all_text)):
            fea_index[all_text[i]]=i
            index_2_label[str(i)]=all_text[i]
    
         
    def represent_instances_fea(self, instances, labels, word_max_len=100, char_max_len=50, onehot=False):
                        
        x_word_list=[]
        x_char_list=[]
        y_list=[]
        data_vocab=[]
        data_oov=[]
        for sentence in instances:
            sentence_list=[]
            sentence_word_list=[]
            label_list=[]
            for j in range(0,len(sentence)):
                word=sentence[j]
                #char fea
                char_list=[0]*char_max_len
                for i in range(len(word[0])):
                    if i<char_max_len:
                        if word[0][i] in self.char_2_index.keys():
                            char_list[i]=self.char_2_index[word[0][i]]
                        else:
                            char_list[i]=self.char_2_index['oov_padding']
                sentence_word_list.append(char_list)
                
                #word fea
                if word[0].lower() in self.word_2_index.keys():
                    sentence_list.append(self.word_2_index[word[0].lower()])
                    if word[0].lower() not in data_vocab:
                        data_vocab.append(word[0].lower())
                        
                else:
                    sentence_list.append(self.word_2_index['sparse_vectors'])
                    if word[0].lower() not in data_oov:
                        data_oov.append(word[0].lower())
                
                #label
                label_list.append(self.label_2_index[word[-1]])
            x_word_list.append(sentence_list)
            x_char_list.append(sentence_word_list)
            y_list.append(label_list)
            
                
        print("dataset vocab:", len(data_vocab),"dataset oov:",len(data_oov))
        
        x_word_np = pad_sequences(x_word_list, word_max_len, value=0, padding='post',truncating='post')  # right padding

        x_char_np = pad_sequences(x_char_list, word_max_len, value=0, padding='post',truncating='post')
        #print(y_list)
        y_np = pad_sequences(y_list, word_max_len, value=0, padding='post',truncating='post')
        
        # y format onehot?
        if onehot: 
            y_np = np.eye(len(labels), dtype='float32')[y_np]
        else:
            y_np = np.expand_dims(y_np, 2)

        return [x_word_np, x_char_np], y_np

if __name__ == '__main__':
    pass
 
            
