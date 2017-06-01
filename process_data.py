## file name: process_data.py
## function: process original data to readable format, 
##           select appropriate label
##           generate processed data file

import sys
import numpy as np
import string
import random

if __name__ == '__main__':

    src_filepath = "./Gene_Chip_Data/microarray.original.txt"
    lab_filepath = "./Gene_Chip_Data/E-TABM-185.sdrf.txt"
    
    # by default, select 7th column
    # Characteristics [DiseaseState]
    lab_idx = 7
    if len(sys.argv) != 1:
        lab_idx = sys.argv[1]

    f = open(lab_filepath, 'r')
    fl = f.readlines()
    lab_len = len(fl)
    lab_name = fl[0].split('\t')[lab_idx]
    print "Select label: ", lab_name
    
    lab_item = []
    lab_item_num = []
    data_lab = []
    min_num = 20 # give up label if data item less than 20
    for i in range(lab_len-1):
        label = fl[i+1].split('\t')[lab_idx]
        data_lab.append(label)
        if label not in lab_item:
            lab_item.append(label)
            lab_item_num.append([])
        
        idx = lab_item.index(label)
        lab_item_num[idx].append(i)
    
    for i in range(len(lab_item_num)):
        lab_item_num[i] = len(lab_item_num[i])

    select_lab = []
    for i in range(len(lab_item)):
        if lab_item_num[i] >= min_num and lab_item[i] != '  ':
            select_lab.append(lab_item[i])
    
    f.close()
    
    # read src data   
    f = open(src_filepath, 'r')
    f.readline()
    data = []
    for i in range(22283):
        line = f.readline().split('\t')[1:]
        line = map(string.atof, line)
        data.append(line)
    
    data = np.array(data).T
    f.close()
    
    # write data
    src_wr = open("./data/data.txt", 'w')
    lab_wr = open("./data/label.txt", 'w')
    ref_wr = open("./data/label_reference.txt", 'w')
    data_num = 0
    for i in range(lab_len-1):
        if data_lab[i] in select_lab:
            for j in range(data.shape[1]):
                src_wr.write(str(data[i][j]))
                if j != data.shape[1]-1:
                    src_wr.write("\t")
            src_wr.write("\n")
            data_num += 1
            lab_wr.write(str(select_lab.index(data_lab[i])))
            lab_wr.write('\n')
    
    for i in range(len(select_lab)):
        ref_wr.write(str(select_lab[i]))
        ref_wr.write("\t")
        ref_wr.write(str(select_lab.index((select_lab[i]))))
        ref_wr.write("\n")
    
    print "Data item number: ", data_num
    print "Label number: ", len(select_lab) 
    
    src_wr.close()
    lab_wr.close()
    ref_wr.close()

    # divide data to train, validate, test
    # 6 : 2 : 2
    f = open("./data/data.txt", 'r')
    test_num = data_num / 5
    vali_num = test_num
    train_num = data_num - 2*test_num
    
    train_list = random.sample(range(0,data_num), train_num)
    test_list = range(0,data_num)
    for obj in train_list:
        test_list.remove(obj)
    vali_list = test_list
    test_list = random.sample(test_list, test_num)
    for obj in test_list:
        vali_list.remove(obj)
    
    train_wr = open("./data/train.txt", 'w')
    vali_wr = open("./data/vali.txt", 'w')
    test_wr = open("./data/test.txt", 'w')
        
    for i in range(data_num):
        fl = f.readline()
        if i in train_list:
            train_wr.write(fl)
        elif i in vali_list:
            vali_wr.write(fl)
        else:
            test_wr.write(fl)

    train_wr.close()
    vali_wr.close()
    test_wr.close()
    
    
    
    
    
    
    
    



