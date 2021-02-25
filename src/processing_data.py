# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:03:03 2021

@author: luol2
"""


import numpy as np

#read ner text (word\tlabel), generate the list[[[w1,label],[w2,label]]]
def ml_intext(file):
    fin=open(file,'r',encoding='utf-8')
    alltexts=fin.read().strip().split('\n\n')
    fin.close()
    data_list=[]
    label_list=[]

    for sents in alltexts:
        lines=sents.split('\n')
        temp_sentece=[]
        for i in range(0,len(lines)):
            seg=lines[i].split('\t')
            temp_sentece.append(seg[:])
            label_list.append(seg[-1])
        
        data_list.append(temp_sentece)
    #print(data_list)
    #print(label_list)
    return data_list,label_list



# model predict result to conll evalute format  [token answer predict]        
def out_BIO(file,raw_pre,raw_input,label_set):
    fout=open(file,'w',encoding='utf-8')
    for i in range(len(raw_input)):
        
        for j in range(len(raw_input[i])):
            if j<len(raw_pre[i]):
                label_id = np.argmax(raw_pre[i][j]) 
                label_tag = label_set[str(label_id)]
            else:
                label_tag='O'
            fout.write(raw_input[i][j][0]+' '+raw_input[i][j][-1]+' '+label_tag+'\n')
        fout.write('\n')
    fout.close()


