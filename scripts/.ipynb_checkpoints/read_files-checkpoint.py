import numpy as np
from Bio import SeqIO
import random

def OneHot(string_18):
    onehot = [0, 0, 0, 0]
    for x in string_18:
        if x == 'A':
            onehot = np.vstack([onehot,[1,0,0,0]])
        if x == 'T':
            onehot = np.vstack([onehot,[0,1,0,0]])
        if x == 'G':
            onehot = np.vstack([onehot,[0,0,1,0]])
        if x == 'C':
            onehot = np.vstack([onehot,[0,0,0,1]])
    onehot = np.delete(onehot,0,0)
    return onehot

def OneHot_new(string_18):
    onehot = np.array([])
    for x in string_18:
        if x == 'A':
            onehot = np.append(onehot,[1,0,0,0])
        if x == 'T':
            onehot = np.append(onehot,[0,1,0,0])
        if x == 'G':
            onehot = np.append(onehot,[0,0,1,0])
        if x == 'C':
            onehot = np.append(onehot,[0,0,0,1])
    return onehot


def getSeq(x):
    """ Will return the sequence from sequenceing folder when given file location
    """
    records = list(SeqIO.parse(x, "fasta"))
    return records 

def getNeg():
    # will return a list of sequence
    seq = getSeq('../data/yeast-upstream-1k-negative.fa')
    pos = getPos()
    # seq[1].seq will access the first sequences
    final_neg = []
    count = 1
    for x in seq:
        print(count)
        count += 1
        y = x.seq
        i = 0
        k = 17
        while (i+k) < len(y):
            temp = y[i:(i+k)]
            if temp in pos:
                i += 1
                print('positive')
            else:
                final_neg += [temp]
                i += 1
    return final_neg
        
def getPos():
    file1 = open('../data/rap1-lieb-positives.txt', 'r') 
    Lines = file1.read().splitlines()
    for x in Lines:
        seq = ''
        for y in x:
            if y == 'A':
                seq += 'T'
            if y == 'T':
                seq += 'A'
            if y == 'G':
                seq += 'C'
            if y == 'C':
                seq += 'G'
        Lines = Lines + [seq]
    return Lines
            
    
def random_sample(list, k):
    fin_list = []
    while x < k:
        
        k += 1
        
