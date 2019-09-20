import pandas as pd
import os
import shutil
import random
import math
import numpy as np
import string
import sys
import json

def fold(current_path,target_path,property_name,num_fold):
    data_total=pd.read_csv(os.path.join(current_path,'train.dat'),sep=' ')
    num_samples=len(data_total.Material)

    data_list=np.arange(1,num_samples+1)
    random.shuffle(data_list)
    batch_size=int(math.ceil(num_samples/num_fold))
    
    try:
        shutil.rmtree(os.path.join(target_path,'%s_cv'%property_name))
    except FileNotFoundError:
        if os.path.exists(target_path)==False:
            os.mkdir(target_path)
        os.mkdir(os.path.join(target_path,'%s_cv'%property_name))
        target_path=os.path.join(target_path,'%s_cv'%property_name)
    finally:
        data_total.to_csv(os.path.join(target_path,'train.dat'),index=False,sep=' ')
        with open(os.path.join(target_path,'cross_validation_info.dat'),'w') as f:
            data={'cross validation type':'%d-fold'%num_fold, 'shuffle data list':data_list.tolist()}
            json.dump(data,f)

    for i in range(0,num_fold):
        try:
            shutil.copytree(current_path,os.path.join(target_path,property_name+'_cv%d'%i))
        except FileExistsError:
            shutil.rmtree(os.path.join(target_path,property_name+'_cv%d'%i))
            shutil.copytree(current_path,os.path.join(target_path,property_name+'_cv%d'%i))
        
        train_list=np.array([])
        for j in range(0,num_fold):
            if i==j:
                val_list=data_list[batch_size*j:min(batch_size*(j+1),num_samples)]
            else:
                train_list=np.append(train_list,data_list[batch_size*j:min(batch_size*(j+1),num_samples)])
        
        data_train=data_total.iloc[train_list-1]
        data_val=data_total.iloc[val_list-1]
        train_list=train_list.astype(int)
        val_list=val_list.astype(int)
        data_train.to_csv(os.path.join(target_path,property_name+'_cv%d'%i,'train.dat'),index=False,sep=' ')
        data_val.to_csv(os.path.join(target_path,property_name+'_cv%d'%i,'validation.dat'),index=False,sep=' ')
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'shuffle.dat'),'w') as f:
            data={'train list':train_list.tolist(), 'validation list':val_list.tolist()}
            json.dump(data,f)
        
        lines=[]
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'SISSO.in'),'r') as f:
            lines=f.readlines()
            for j in range(len(lines)):
                if lines[j].startswith('nsample'):
                    lines[j]='nsample=%d'%train_list.shape[0]+'\t! number of samples for each task (seperate the numbers by comma for ntask >1)\n'
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'SISSO.in'),'w') as f:
            f.writelines(lines)



def leave_out(current_path,target_path,property_name,num_iter,num_out=0,frac=0):
    if num_out and frac:
        print("Please input one of num_out and frac!")
        return None
    
    data_total=pd.read_csv(os.path.join(current_path,'train.dat'),sep=' ')
    num_samples=len(data_total.Material)
    
    try:
        shutil.rmtree(os.path.join(target_path,'%s_cv'%property_name))
    except FileNotFoundError:
        if os.path.exists(target_path)==False:
            os.mkdir(target_path)
        os.mkdir(os.path.join(target_path,'%s_cv'%property_name))
        target_path=os.path.join(target_path,'%s_cv'%property_name)
    finally:
        data_total.to_csv(os.path.join(target_path,'train.dat'),index=False,sep=' ')
        with open(os.path.join(target_path,'cross_validation_info.dat'),'w') as f:
            if num_out:
                data={'cross validation type':'leave-%d-out'%num_out, 'iteration times':num_iter}
                json.dump(data,f)
            else:
                data={'cross validation type':'leave-%d%%-out'%int(frac*100), 'iteration times':num_iter}
                json.dump(data,f)

    for i in range(0,num_iter):
        try:
            shutil.copytree(current_path,os.path.join(target_path,property_name+'_cv%d'%i))
        except FileExistsError:
            shutil.rmtree(os.path.join(target_path,property_name+'_cv%d'%i))
            shutil.copytree(current_path,os.path.join(target_path,property_name+'_cv%d'%i))
        
        data_list=list(range(1,num_samples+1))
        if num_out==0 and frac:
            num_out=int(num_samples*frac)
        val_list=random.sample(data_list,num_out)
        train_list=[x for x in data_list if x not in val_list]
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'shuffle.dat'),'w') as f:
            data={'train list':list(train_list), 'validation list':list(val_list)}
            json.dump(data,f)
        train_list=np.array(train_list)
        val_list=np.array(val_list)
        data_train=data_total.loc[train_list-1]
        data_val=data_total.loc[val_list-1]
        data_train.to_csv(os.path.join(target_path,property_name+'_cv%d'%i,'train.dat'),index=False,sep=' ')
        data_val.to_csv(os.path.join(target_path,property_name+'_cv%d'%i,'validation.dat'),index=False,sep=' ')
        
        
        
        lines=[]
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'SISSO.in'),'r') as f:
            lines=f.readlines()
            for j in range(len(lines)):
                if lines[j].startswith('nsample'):
                    lines[j]='nsample=%d'%train_list.shape[0]+'\t! number of samples for each task (seperate the numbers by comma for ntask >1)\n'
        with open(os.path.join(target_path,property_name+'_cv%d'%i,'SISSO.in'),'w') as f:
            f.writelines(lines)