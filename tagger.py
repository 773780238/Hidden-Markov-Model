import numpy as np

from util import accuracy
from hmm import HMM
def normalize(v):
    if v.ndim == 1:
        norm = sum(v)
    else:
        norm = sum(sum(v))
    if norm == 0: 
       return v
    return v / norm
# TODO:
def model_training(train_data, tags):
    #####lowercased######
    for data in train_data:       
        for it in range(data.length):
            data.words[it] = data.words[it].lower()
            
    #####################
    S = len(tags)
    pi = np.zeros(S)
    A = np.zeros([S,S])
    B = []
    Bc= np.zeros([S,1])
    Ac= np.zeros([S,S])
    obs_dict = {}
    states_symbols = {}
    for i in range(S):
        if not tags[i] in states_symbols.keys():
            states_symbols[tags[i]] = i
    numS = np.zeros(S)
    num1S= np.zeros(S)
    ####################################
    for data in train_data:
        firsttag = data.tags[0]
        num1S[states_symbols[firsttag]] += 1
        for i in range(data.length):
            word = data.words[i]
            tag = data.tags[i]
            if not word  in obs_dict.keys():
                obs_dict[word] = len(obs_dict)
                Bc = np.append(Bc,np.zeros([S,1]),axis = 1)
            Bc[states_symbols[tag],obs_dict[word]] += 1
            
            numS[states_symbols[tag]] += 1
            if i != data.length-1:
                Ac[states_symbols[tag],states_symbols[ data.tags[i+1]]] += 1
    B = np.zeros(np.shape(Bc))            
    pi = normalize(num1S)                
    for s in range(S):
        for sp in range(S):
            if numS[s] == 0:
                A[s,sp] = 0
            else:
                A[s,sp] = Ac[s,sp]/numS[s]
    for s in range(len(Bc)):
        for o in range(len(Bc[0])):
            if numS[s] == 0:
                B[s,o] = 0
            else:
                B[s,o] = Bc[s,o]/numS[s]
    ###################################

        model = HMM(pi, A, B, obs_dict, states_symbols)       
  
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    taggings = []
    S = len(tags)
    #####lowercased######
    for data in test_data:       
        for it in range(data.length):
            data.words[it] = data.words[it].lower()
            
    #####################
    for test in test_data:
        ####expand####
        for word in test.words:
            if not word in model.obs_dict.keys():
                model.obs_dict[word] = len(model.obs_dict)
                A = np.zeros([S,1])
                for i in range(S):
                    A[i,0] = 1e-6
                model.B = np.append(model.B,A,axis=1)
               
        ##############
        tagging = model.viterbi(test.words)
        taggings.append(tagging)

    
    
    
    
    return taggings
    
