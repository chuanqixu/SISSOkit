import pandas as pd
import os
import shutil
import random
import math
import numpy as np
import string
import sys
import json
import re

def fold(current_path,target_path,property_name,num_fold):
    with open(os.path.join(current_path,'SISSO.in'),'r') as f:
            input_file=f.read()
            task_number=int(re.findall(r'ntask\s*=\s*(\d+)',input_file)[0])
            samples_number=re.findall(r'nsample\s*=\s*([\d,]+)',input_file)[0]
            samples_number=re.split(r'[, ]+',samples_number)
            samples_number=list(map(int,samples_number))
    
    data_total=pd.read_csv(os.path.join(current_path,'train.dat'),sep=' ')
    
    i=1
    data_list=[]
    for sample_num in samples_number:
        data_list.append(list(range(i,i+sample_num)))
        i+=sample_num
    
    for task in range(0,task_number):
        random.shuffle(data_list[task])
    batch_size=list(map(lambda x: int(math.ceil(x/num_fold)), samples_number))
    
    try:
        if os.path.exists(os.path.join(target_path,'%s_cv'%property_name)):
            print('Directory already exists.\nDo you want to remove the directory?')
            a=input('y|n\n')
            if a=='y':
                shutil.rmtree(os.path.join(target_path,'%s_cv'%property_name))
            if a=='n':
                print('Please input a new target path!')
                return None
    except FileNotFoundError:
        if os.path.exists(target_path)==False:
            os.mkdir(target_path)
    finally:
        os.mkdir(os.path.join(target_path,'%s_cv'%property_name))
        target_path=os.path.join(target_path,'%s_cv'%property_name)
        data_total.to_csv(os.path.join(target_path,'train.dat'),index=False,sep=' ')
        with open(os.path.join(target_path,'cross_validation_info.dat'),'w') as f:
            json.dump({'cross_validation_type':'%d-fold'%num_fold,'shuffle_data_list':data_list},f)

    for i in range(0,num_fold):
        try:
            shutil.copytree(current_path,os.path.join(target_path,property_name+'_cv%d'%i))
        except FileExistsError:
            shutil.rmtree(os.path.join(target_path,property_name+'_cv%d'%i))
            shutil.copytree(current_path,os.path.join(target_path,property_name+'_cv%d'%i))
        
        
        val_list=[]
        train_list=[]
        for task in range(0,task_number):
            train_list_t=[]
            if batch_size[task]*i<samples_number[task]:
                for j in range(0,num_fold):
                    if batch_size[task]*j<samples_number[task]:
                        if i==j:
                            val_list.append(data_list[task][batch_size[task]*j:min(batch_size[task]*(j+1),samples_number[task])])
                        else:
                            train_list_t.append(data_list[task][batch_size[task]*j:min(batch_size[task]*(j+1),samples_number[task])])
                    else:
                        break
            else:
                val_list.append([])
                train_list_t.append(data_list[task])
            train_list.append(np.hstack(train_list_t).tolist())
        
        train_len=list(map(len,train_list))
        val_len=list(map(len,val_list))
        
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'shuffle.dat'),'w') as f:
            json.dump({'training_list':train_list,'training_samples_number':train_len,'validation_list':val_list,'validation_samples_number':val_len},f)
        
        data_train=data_total.iloc[np.hstack(train_list)-1]
        data_val=data_total.iloc[np.hstack(val_list)-1]
        data_train.to_csv(os.path.join(target_path,property_name+'_cv%d'%i,'train.dat'),index=False,sep=' ')
        data_val.to_csv(os.path.join(target_path,property_name+'_cv%d'%i,'validation.dat'),index=False,sep=' ')
        
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'SISSO.in'),'r') as f:
            lines=f.readlines()
            for j in range(len(lines)):
                if lines[j].startswith('nsample'):
                    lines[j]='nsample=%s'%(str(train_len).strip('[]'))+'\t! number of samples for each task (seperate the numbers by comma for ntask >1)\n'
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'SISSO.in'),'w') as f:
            f.writelines(lines)



def leave_out(current_path,target_path,property_name,num_iter,frac=0,num_out=0):
    if num_out and frac:
        print("Please input one of num_out and frac!")
        return None
    
    with open(os.path.join(current_path,'SISSO.in'),'r') as f:
            input_file=f.read()
            task_number=int(re.findall(r'ntask\s*=\s*(\d+)',input_file)[0])
            samples_number=re.findall(r'nsample\s*=\s*([\d,]+)',input_file)[0]
            samples_number=re.split(r'[, ]+',samples_number)
            samples_number=list(map(int,samples_number))
    
    data_total=pd.read_csv(os.path.join(current_path,'train.dat'),sep=' ')
    
    i=1
    data_list=[]
    for sample_num in samples_number:
        data_list.append(list(range(i,i+sample_num)))
        i+=sample_num
    
    try:
        if os.path.exists(os.path.join(target_path,'%s_cv'%property_name)):
            print('Directory already exists.\nDo you want to remove the directory?')
            a=input('y|n\n')
            if a=='y':
                shutil.rmtree(os.path.join(target_path,'%s_cv'%property_name))
            if a=='n':
                print('Please input a new target path!')
                return None
    except FileNotFoundError:
        if os.path.exists(target_path)==False:
            os.mkdir(target_path)
    finally:
        os.mkdir(os.path.join(target_path,'%s_cv'%property_name))
        target_path=os.path.join(target_path,'%s_cv'%property_name)
        data_total.to_csv(os.path.join(target_path,'train.dat'),index=False,sep=' ')
        with open(os.path.join(target_path,'cross_validation_info.dat'),'w') as f:
            if num_out:
                json.dump({'cross_validation_type':'leave-%d-out'%num_out,'iteration_times':num_iter},f)
            else:
                json.dump({'cross_validation_type':'leave-%d%%-out'%int(frac*100),'iteration_times':num_iter},f)

    num_out=[]
    total_samples_number=0
    for task in range(task_number):
        total_samples_number+=samples_number[task]
    if num_out:
        frac=num_out/total_samples_number
    
    if frac:
        for task in range(0,task_number):
            num_out.append(round(samples_number[task]*frac))
    
    for i in range(0,num_iter):
        try:
            shutil.copytree(current_path,os.path.join(target_path,property_name+'_cv%d'%i))
        except FileExistsError:
            shutil.rmtree(os.path.join(target_path,property_name+'_cv%d'%i))
            shutil.copytree(current_path,os.path.join(target_path,property_name+'_cv%d'%i))
        
        val_list=[]
        for task in range(0,task_number):
            val_list.append(random.sample(data_list[task],num_out[task]))
        train_list=[[x for x in data_list[task] if x not in val_list[task]] for task in range(0,task_number) ]
        train_len=list(map(len,train_list))
        val_len=list(map(len,val_list))
        data_train=data_total.iloc[np.hstack(train_list)-1]
        data_val=data_total.iloc[np.hstack(val_list)-1]
        data_train.to_csv(os.path.join(target_path,property_name+'_cv%d'%i,'train.dat'),index=False,sep=' ')
        data_val.to_csv(os.path.join(target_path,property_name+'_cv%d'%i,'validation.dat'),index=False,sep=' ')
        
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'shuffle.dat'),'w') as f:
            json.dump({'training_list':train_list,'training_samples_number':train_len,'validation_list':val_list,'validation_samples_number':val_len},f)
        
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'SISSO.in'),'r') as f:
            lines=f.readlines()
            for j in range(len(lines)):
                if lines[j].startswith('nsample'):
                    lines[j]='nsample=%s'%(str(train_len).strip('[]'))+'\t! number of samples for each task (seperate the numbers by comma for ntask >1)\n'
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'SISSO.in'),'w') as f:
            f.writelines(lines)