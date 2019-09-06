import pandas as pd
import numpy as np
import string
import re
import os
import sys
import functools
import math
import json

def evaluate_expression(expression,data):
    """
    Return the value computed using given expression over data.
    """
    operators=['+','-','*','/','exp','exp-','^-1','^2','^3','sqrt','cbrt','log','abs','^6','sin','cos','(',')']
    OPTR=[]
    OPND=[]
    s=''
    i=0
    while i < len(expression):
        if re.match(r'\w',expression[i]):
            s+=expression[i]
        elif expression[i]=='-' and expression[i-1]=='p':
            s+=expression[i]
        else:
            if s:
                if s in operators:
                    OPTR.append(s)
                else:
                    OPND.append(data[s])
                s=''
            OPTR.append(expression[i])
            if expression[i]==')':
                OPTR.pop()
                if i+1<len(expression) and expression[i+1]=='^':
                    pattern=re.compile(r'\^-?\d')
                    power=pattern.match(expression,i+1).group()
                    i+=len(power)
                    operand=OPND.pop()
                    if power=='^-1':
                        OPND.append(np.power(operand,-1,dtype=np.float64))
                    elif power=='^2':
                        OPND.append(np.power(operand,2,dtype=np.float64))
                    elif power=='^3':
                        OPND.append(np.power(operand,3,dtype=np.float64))
                    elif power=='^6':
                        OPND.append(np.power(operand,6,dtype=np.float64))
                    OPTR.pop()
                elif len(OPTR)>1 and OPTR[-1]=='(':
                    operand=OPND.pop()
                    if OPTR[-2]=='exp':
                        OPND.append(np.exp(operand,dtype=np.float64))
                    elif OPTR[-2]=='exp-':
                        OPND.append(np.exp(-operand,dtype=np.float64))
                    elif OPTR[-2]=='sqrt':
                        OPND.append(np.sqrt(operand,dtype=np.float64))
                    elif OPTR[-2]=='cbrt':
                        OPND.append(np.cbrt(operand,dtype=np.float64))
                    elif OPTR[-2]=='log':
                        OPND.append(np.log(operand,dtype=np.float64))
                    elif OPTR[-2]=='abs':
                        OPND.append(np.abs(operand,dtype=np.float64))
                    elif OPTR[-2]=='sin':
                        OPND.append(np.sin(operand,dtype=np.float64))
                    elif OPTR[-2]=='cos':
                        OPND.append(np.cos(operand,dtype=np.float64))
                    OPTR.pop()
                    OPTR.pop()
                elif len(OPTR)==1 and OPTR[0]=='(':
                    pass
                else:
                    operand2=OPND.pop()
                    operand1=OPND.pop()
                    operator=OPTR.pop()
                    OPTR.pop()
                    if operator=='+':
                        OPND.append(operand1+operand2)
                    elif operator=='-':
                        if OPTR[-1]=='abs':
                            OPND.append(np.abs(operand1-operand2))
                            OPTR.pop()
                        else:
                            OPND.append(operand1-operand2)
                    elif operator=='*':
                        OPND.append(operand1*operand2)
                    elif operator=='/':
                        OPND.append(operand1/operand2)
        i+=1
    return OPND.pop()


def compute_using_descriptors(path=None,result=None,training=True,data=None,task_idx=None,dimension_idx=None):
    """
    Return a list [task_index], whose item is numpy array [dimension, sample_index],
    whose item is the value computed using descriptors found by SISSO.
    """
    if path:
        result=Result(path)
    pred=[]
    if training==True:
        for task in range(0,result.task_number):
            pred_t=[]
            for dimension in range(0,result.dimension):
                value=0
                for i in range(0,dimension+1):
                    value+=result.coefficients()[task][dimension][i]*evaluate_expression(result.descriptors()[dimension][i],result.data_task[task])
                value+=result.intercepts()[task][dimension]
                pred_t.append(list(value))
            pred.append(np.array(pred_t))
    if training==False:
        if data==None:
            for task in range(0,result.task_number):
                pred_t=[]
                for dimension in range(0,result.dimension):
                    value=0
                    for i in range(0,dimension+1):
                        value+=result.coefficients()[task][dimension][i]*evaluate_expression(result.descriptors()[dimension][i],result.validation_data_task[task])
                    value+=result.intercepts()[task][dimension]
                    pred_t.append(list(value))
                pred.append(np.array(pred_t))
        else:
            if isinstance(data,str):
                data=pd.read_csv(os.path.join(data),sep=' ')
            else:
                for i in range(0,dimension_idx):
                    value+=result.coefficients()[task_idx][dimension_idx][i]*evaluate_expression(result.descriptors()[dimension_idx][i],data)
                value+=result.intercepts()[task_idx][dimension_idx]
                pred=value
    return pred


def compute_errors(errors,samples_number=[]):
    """
    Return the errors of given numpy array errors (task_index, dimension, sample_index), if errors is 2D numpy array,
    or return the errors of given 1D numpy array error
    """
    # 1D [sample]
    if isinstance(errors[0],float):
        errors=np.sort(np.abs(errors))
        error=np.array([np.sqrt(np.mean(np.power(errors,2))),
                                np.mean(errors),
                                np.percentile(errors,25),
                                np.percentile(errors,50),
                                np.percentile(errors,75),
                                np.percentile(errors,95),
                                errors[-1]])
        error=pd.Series(error,
                        index=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'])
    # 3D [task, dimension, sample] or [cv, dimension, sample]
    elif isinstance(errors,list) and isinstance(errors[0],np.ndarray):
        task_num=len(errors)
        dimension_num=len(errors[0])
        error=[]
        for task in range(task_num):
            error.append(pd.DataFrame([compute_errors(errors[task][dimension]) for dimension in range(dimension_num)],
                                    columns=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],
                                    index=list(range(1,dimension_num+1))))
    # 4D [cv, task, dimension, sample]
    elif isinstance(errors,list) and isinstance(errors[0],list):
        cv_num=len(errors)
        task_num=len(errors[0])
        dimension_num=len(errors[0][0])
        error=[]
        for cv in range(cv_num):
            error_cv=[]
            for task in range(task_num):
                error_cv.append(pd.DataFrame([compute_errors(errors[cv][task][dimension]) for dimension in range(dimension_num)],
                                            columns=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],
                                            index=list(range(1,dimension_num+1))))
            error.append(error_cv)
    # 2D [dimension, sample]
    elif errors.ndim==2:
        dimension_num=len(errors)
        error=pd.DataFrame([compute_errors(errors[dimension]) for dimension in range(dimension_num)],
                            columns=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],
                            index=list(range(1,dimension_num+1)))
    return error






class Result(object):
    """
    Evaluate the SISSO results.
    """
    def __init__(self,current_path):
        self.current_path=current_path
        with open(os.path.join(self.current_path,'SISSO.in'),'r') as f:
            input_file=f.read()
            subs_sis=int(re.findall(r'subs_sis\s*=\s*(\d+)',input_file)[0])
            rung=int(re.findall(r'rung\s*=\s*(\d+)',input_file)[0])
            dimension=int(re.findall(r'desc_dim\s*=\s*(\d+)',input_file)[0])
            operation_set=re.findall(r"opset\s*=\s*'(.+)'",input_file)
            operation_set=re.split(r'[\(\)]+',operation_set[0])[1:-1]
            task_number=int(re.findall(r'ntask\s*=\s*(\d+)',input_file)[0])
            samples_number=re.findall(r'nsample\s*=\s*([\d, ]+)',input_file)[0]
            samples_number=re.split(r'[,\s]+',samples_number)
            if samples_number[-1]=='':
                samples_number=samples_number[:-1]
            samples_number=list(map(int,samples_number))
            task_weighting=int(re.findall(r'task_weighting\s*=\s*(\d+)',input_file)[0])
        self.task_weighting=task_weighting
        self.task_number=task_number
        self.operation_set=operation_set
        self.subs_sis=subs_sis
        self.rung=rung
        self.dimension=dimension
        self.samples_number=samples_number
        self.data=pd.read_csv(os.path.join(current_path,'train.dat'),sep=' ')
        self.property_name=self.data.columns.tolist()[1]
        self.property=self.data.iloc[:,1]
        self.features_name=self.data.columns.tolist()[2:]
        self.materials=self.data.iloc[:,0]
        
        self.data_task=[]
        self.property_task=[]
        self.materials_task=[]
        i=0
        for task in range(0,self.task_number):
            self.data_task.append(self.data.iloc[i:i+self.samples_number[task]])
            self.property_task.append(self.property.iloc[i:i+self.samples_number[task]])
            self.materials_task.append(self.materials.iloc[i:i+self.samples_number[task]])
            i+=self.samples_number[task]
        
        if os.path.exists(os.path.join(self.current_path,'validation.dat')):
            self.validation_data=pd.read_csv(os.path.join(current_path,'validation.dat'),sep=' ')
            with open(os.path.join(self.current_path,'shuffle.dat'),'r') as f:
                shuffle=json.load(f)
            self.samples_number=shuffle['training_samples_number']
            self.validation_samples_number=shuffle['validation_samples_number']
            self.validation_data_task=[]
            i=0
            for task in range(0,self.task_number):
                self.validation_data_task.append(self.validation_data.iloc[i:i+self.validation_samples_number[task]])
                i+=self.validation_samples_number[task]

    def __repr__(self):
        text='#'*50+'\n'+'Result of SISSO\n'+'#'*50
        text+='\nProperty Name: %s\nTask Number: %d\nRung: %d\nDimension: %d\nSubs_sis: %d\n'%(self.property_name,self.task_number,self.rung,self.dimension,self.subs_sis)
        return text
    
    def baseline(self):
        errors=np.sort((self.property-self.property.mean()).abs().values)
        errors=np.array([self.property.mean(),
                        self.property.std(),
                        np.mean(errors),
                        np.percentile(errors,25),
                        np.percentile(errors,50),
                        np.percentile(errors,75),
                        np.percentile(errors,95),
                        errors[-1]])
        return pd.Series(errors,index=['mean','std','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'])
    
    def descriptors(self,path=None):
        """
        Return a list, whose ith item is ith D descriptors.
        """
        if path==None:
            path=self.current_path
        descriptors_all=[]
        with open(os.path.join(path,'SISSO.out'),'r') as f:
            input_file=f.read()
            descriptors_total=re.findall(r'descriptor:[\s\S]*?coefficients',input_file)
            for dimension in range(0,self.dimension):
                descriptors_d=descriptors_total[dimension]
                descriptors_d=re.split(r'\s+',descriptors_d)
                descriptors_d=descriptors_d[1:dimension+2]
                descriptors_d=[x[1] for x in list(map(lambda x: re.split(r':',x),descriptors_d))]
                descriptors_d=list(map(lambda x: x.replace(r'[',r'('),descriptors_d))
                descriptors_d=list(map(lambda x: x.replace(r']',r')'),descriptors_d))
                descriptors_all.append(descriptors_d)
        return descriptors_all
    
    def coefficients(self,path=None):
        """
        Return a list [task_index, dimension, descriptor_index]
        """
        if path==None:
            path=self.current_path
        coefficients_all=[]
        for task in range(0,self.task_number):
            coefficients_t=[]
            with open(os.path.join(path,'SISSO.out'),'r') as f:
                input_file=f.read()
                coefficients_total=re.findall(r'coefficients_00%d:(.*)'%(task+1),input_file)
                for dimension in range(0,self.dimension):
                    coefficients_d=re.split(r'\s+',coefficients_total[dimension])[1:]
                    coefficients_d=list(map(float,coefficients_d))
                    coefficients_t.append(coefficients_d)
            coefficients_all.append(coefficients_t)
        return coefficients_all
                
    
    def intercepts(self,path=None):
        """
        Return a list [task_index, dimension]
        """
        if path==None:
            path=self.current_path
        intercepts_all=[]
        for task in range(0,self.task_number):
            with open(os.path.join(path,'SISSO.out'),'r') as f:
                input_file=f.read()
                intercepts_t=re.findall(r'Intercept_00%d:(.*)'%(task+1),input_file)
                intercepts_t=list(map(float,intercepts_t))
            intercepts_all.append(intercepts_t)
        return intercepts_all
    
    def features_percent(self):
        """
        Compute the percentages of each feature in top subs_sis 1D descriptors.
        """
        """
        feature_space=pd.read_csv(os.path.join(self.current_path,'feature_space','Uspace.name'),sep=' ',header=None).iloc[0:self.subs_sis,0]
        feature_percent=pd.DataFrame(columns=('feature','percent'))
        index=0
        for feature_name in self.features_name:
            percent=feature_space.str.contains(feature_name).sum()/self.subs_sis
            feature_percent.loc[index]={'feature':feature_name,'percent':percent}
            index+=1
        feature_percent.sort_values('percent',inplace=True,ascending=False)
        return feature_percent
        """
        feature_space=pd.read_csv(os.path.join(self.current_path,'feature_space','Uspace.name'),sep=' ',header=None).iloc[0:self.subs_sis,0]
        feature_percent=pd.DataFrame(columns=self.features_name,index=('percent',))
        for feature_name in self.features_name:
            percent=feature_space.str.contains(feature_name).sum()/self.subs_sis
            feature_percent.loc['percent',feature_name]=percent
        return feature_percent
    
    def evaluate_expression(self,expression,data=None):
        """
        Return the value computed using given expression over data.
        """
        return evaluate_expression(expression,self.data)
    
    def values(self,training=True,display_task=False):
        """
        Return a 2D numpy array [dimension, sample_index],
        whose item is the value computed using descriptors found by SISSO.
        
        Return numpy array (task_index, sample_index, dimension),
        """
        if display_task==True:
            return compute_using_descriptors(result=self,training=training)
        else:
            return np.hstack(compute_using_descriptors(result=self,training=training))
    
    def errors(self,training=True,display_task=False):
        """
        Return a numpy array [task, dimension, sample_index], whose value is error.
        
        Return a numpy array [dimension, sample_index], whose value is error.
        """
        if display_task==True:
            pred=self.values(training=training,display_task=True)
            return [pred[task]-self.property_task[task].values for task in range(0,self.task_number)]
        else:
            if training==True:
                return self.values(training=training,display_task=False)-self.property.values
            else:
                return self.values(training=training,display_task=False)-self.validation_data.iloc[:,1].values
    
    def total_errors(self,training=True,display_task=False):
        """
        Return a list [task_index], whose item is a pandas DataFrame [dimension, type of error]
        
        Return a pandas DataFrame [dimension, type of error].
        """
        return compute_errors(self.errors(training=training,display_task=display_task))




class Results(Result):
    """
    Evaluate the cross validation results of SISSO.
    """
    
    def __init__(self,current_path,property_name,drop_index=[]):
        self.current_path=current_path
        cv_names=sorted(list(filter(lambda x: x.startswith(property_name+'_cv'),os.listdir(current_path))),
                        key=lambda x:int(x.split(property_name+'_cv')[-1]))
        cv_number=len(cv_names)
        dir_list=list(map(lambda cv_name:os.path.join(current_path,cv_name),cv_names))
        self.cv_path=[dir_list[cv] for cv in range(cv_number) if cv not in drop_index]
        self.property_name=property_name
        self.drop_index=drop_index
        self.cv_number=cv_number-len(drop_index)
        self.total_data=pd.read_csv(os.path.join(current_path,'train.dat'),sep=' ')
        
        with open(os.path.join(self.current_path,'cross_validation_info.dat'),'r') as f:
            cv_info=json.load(f)
        self.cross_validation_type=cv_info['cross_validation_type']
        
        with open(os.path.join(self.cv_path[0],'SISSO.in'),'r') as f:
            input_file=f.read()
            subs_sis=int(re.findall(r'subs_sis\s*=\s*(\d+)',input_file)[0])
            rung=int(re.findall(r'rung\s*=\s*(\d+)',input_file)[0])
            dimension=int(re.findall(r'desc_dim\s*=\s*(\d+)',input_file)[0])
            operation_set=re.findall(r"opset\s*=\s*'(.+)'",input_file)
            operation_set=re.split(r'[\(\)]+',operation_set[0])[1:-1]
            task_number=int(re.findall(r'ntask\s*=\s*(\d+)',input_file)[0])
            task_weighting=int(re.findall(r'task_weighting\s*=\s*(\d+)',input_file)[0])
        self.task_number=task_number
        self.task_weighting=task_weighting
        self.operation_set=operation_set
        self.subs_sis=subs_sis
        self.rung=rung
        self.dimension=dimension
        self.total_materials_number=len(pd.read_csv(os.path.join(current_path,'train.dat'),sep=' '))
        self.data=[]
        self.materials=[]
        self.samples_number=[]
        self.validation_samples_number=[]
        self.property=[]
        self.validation_data=[]
        for cv_path in self.cv_path:
            data=pd.read_csv(os.path.join(cv_path,'train.dat'),sep=' ')
            self.data.append(data)
            self.validation_data.append(pd.read_csv(os.path.join(cv_path,'validation.dat'),sep=' '))
            self.property.append(data.iloc[:,1])
            self.materials.append(data.iloc[:,0])
            with open(os.path.join(cv_path,'shuffle.dat'),'r') as f:
                shuffle=json.load(f)
            self.samples_number.append(shuffle['training_samples_number'])
            self.validation_samples_number.append(shuffle['validation_samples_number'])
        self.features_name=self.data[0].columns.tolist()[2:]
        self.samples_number=np.array(self.samples_number)
        self.validation_samples_number=np.array(self.validation_samples_number)
        
        self.data_task=[]
        self.property_task=[]
        self.materials_task=[]
        self.validation_data_task=[]
        for cv in range(0,self.cv_number):
            data_t=[]
            property_t=[]
            materials_t=[]
            validation_data_t=[]
            i,j=0,0
            for task in range(0,self.task_number):
                data_t.append(self.data[cv].iloc[i:i+self.samples_number[cv,task]])
                validation_data_t.append(self.validation_data[cv].iloc[j:j+self.validation_samples_number[cv,task]])
                property_t.append(self.property[cv].iloc[i:i+self.samples_number[cv,task]])
                materials_t.append(self.materials[cv].iloc[i:i+self.samples_number[cv,task]])
                i+=self.samples_number[cv,task]
                j+=self.validation_samples_number[cv,task]
            self.data_task.append(data_t)
            self.validation_data_task.append(validation_data_t)
            self.property_task.append(property_t)
            self.materials_task.append(materials_t)
        self.property_task=np.array(self.property_task)
        self.materials_task=np.array(self.materials_task)
    
    def __getitem__(self,index):
        if isinstance(index,slice):
            return [Result(self.cv_path[i]) for i in range(index.start,index.stop)]
        else:
            return Result(self.cv_path[index])
    
    def __repr__(self):
        text='#'*50+'\n'+'Cross Validation Results of SISSO\n'+'#'*50
        with open(os.path.join(self.current_path,'cross_validation_info.dat'),'r') as f:
            cv_info=json.load(f)
        if 'shuffle_data_list' in cv_info:
            text+=('\nCross Validation Type: %s\nShuffle Data List: '%cv_info['cross_validation_type']+str(cv_info['shuffle_data_list']))
        else:
            text+=('\nCross Validation Type: %s\nIteration: '%cv_info['cross_validation_type']+str(self.cv_number))
        text+='\nProperty Name: %s\nTask Number: %d\nRung: %d\nDimension: %d\nSubs_sis: %d'%(self.property_name,self.task_number,self.rung,self.dimension,self.subs_sis)
        return text
    
    def baseline(self):
        total_property=self.total_data.iloc[:,1]
        errors=np.sort((total_property-total_property.mean()).abs().values)
        errors=np.array([total_property.mean(),
                        total_property.std(),
                        np.mean(errors),
                        np.percentile(errors,25),
                        np.percentile(errors,50),
                        np.percentile(errors,75),
                        np.percentile(errors,95),
                        errors[-1]])
        return pd.Series(errors,index=['mean','std','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'])
    
    def find_materials_in_validation(self,*idxs):
        if self.cross_validation_type.startswith('leave'):
            val_num=len(self.validation_data[0])
            return ([self.validation_data[int(index/val_num)].iloc[index%val_num,0] for index in idxs],
                    [int(self.cv_path[int(index/val_num)].split('cv')[-1]) for index in idxs])
    
    def descriptors(self):
        """
        Return a list whose index refers to the index of corss validation.
        Each item in the list is also a list, whose jth item is j+1 D descriptors.
        """
        return [super(Results,self).descriptors(path=cv_path) for cv_path in self.cv_path]
    
    def coefficients(self):
        """
        Return a list whose index refers to the index of corss validation.
        Each item in the list is also a list, whose jth item is j+1 D coefficients.
        """
        return [super(Results,self).coefficients(path=cv_path) for cv_path in self.cv_path]
    
    def intercepts(self):
        """
        Return a list whose index refers to the index of corss validation.
        Each item in the list is also a list, whose jth item is j+1 D intercepts.
        """
        return [super(Results,self).intercepts(path=cv_path) for cv_path in self.cv_path]
    
    def features_percent(self):
        """
        Return the percent of each feature in the top subs_sis descriptors.
        There are total cv_number*subs_sis descriptors,
        the feature percent is the percent over these descriptors.
        """
        feature_percent=pd.DataFrame(columns=self.features_name,index=('percent',))
        feature_percent.iloc[0,:]=0
        for cv_path in self.cv_path:
            feature_space=pd.read_csv(os.path.join(cv_path,'feature_space','Uspace.name'),sep=' ',header=None).iloc[0:self.subs_sis,0]
            for feature_name in self.features_name:
                count=feature_space.str.contains(feature_name).sum()
                feature_percent.loc['percent',feature_name]+=count
        feature_percent.iloc[0,:]=feature_percent.iloc[0,:]/(self.cv_number*self.subs_sis)
        return feature_percent
    
    def descriptor_percent(self,descriptor):
        """
        Return the percent of given descriptor appearing in the cross validation top subs_sis descriptors,
        and return the appearing index in the descriptor space.
        """
        count=0
        descriptor_index=np.zeros(self.cv_number)
        for cv in range(0,self.cv_number):
            feature_space=pd.read_csv(os.path.join(self.cv_path[cv],'feature_space','Uspace.name'),sep=' ',header=None).iloc[0:self.subs_sis,0]
            try:
                descriptor_index[cv]=feature_space.tolist().index(descriptor)+1
                count+=1
            except ValueError:
                descriptor_index[cv]=None
        return count/self.cv_number,descriptor_index
    
    def values(self,training=True,display_cv=False,display_task=False):
        """
        Return a list [cv_index],
        whose item is a list [task_index], whose item is a 2D numpy array [dimension, sample_index, dimension].
        
        Return a list [cv_index], whose item is a 2D numpy array [dimension, sample_index],
        whose item is the value computed using descriptors found by SISSO.
        """
        if display_cv==True:
            if display_task==True:
                return [compute_using_descriptors(path=cv_path,
                                        training=training)
                for cv_path in self.cv_path]
            else:
                return [np.hstack(compute_using_descriptors(path=cv_path,
                                        training=training))
                for cv_path in self.cv_path]
        else:
            return np.hstack(self.values(training=training,display_cv=True,display_task=False))
        """
        if display_cv==True:
            if training==True:
                if display_task==True:
                    return [compute_using_descriptors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)))
                            for cv in range(0,self.cv_number)]
                else:
                    return [np.hstack(compute_using_descriptors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv))))
                    for cv in range(0,self.cv_number)]
            else:
                if display_task==True:
                    return [compute_using_descriptors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)),
                                            training=False)
                    for cv in range(0,self.cv_number)]
                else:
                    return [np.hstack(compute_using_descriptors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)),
                                            training=False))
                    for cv in range(0,self.cv_number)]
        else:
            return np.hstack(self.values(training=training,display_cv=True,display_task=False))
        """
    
    def errors(self,training=True,display_cv=False,display_task=False):
        """
        Return a list [cv_index, task_index],
        whose item is a 2D numpy array [dimension, sample_index].
        
        Return a list [cv_index],
        whose item is a 2D numpy array [dimension, sample_index].
        """
        if display_cv:
            if training:
                if display_task:
                    error=[]
                    pred=self.values(training=True,display_cv=True,display_task=True)
                    for cv in range(0,self.cv_number):
                        error_cv=[]
                        for task in range(0,self.task_number):
                            error_cv.append(pred[cv][task]-self.property_task[cv][task])
                        error.append(error_cv)
                    return error
                else:
                    pred=self.values(training=True,display_cv=True,display_task=True)
                    return [np.hstack(pred[cv])-np.hstack(self.property_task[cv])
                            for cv in range(0,self.cv_number)]
            else:
                if display_task:
                    error=[]
                    pred=self.values(training=False,display_cv=True,display_task=True)
                    for cv in range(0,self.cv_number):
                        error_cv=[]
                        for task in range(0,self.task_number):
                            error_cv.append(pred[cv][task]-self.validation_data_task[cv][task].iloc[:,1].values)
                        error.append(error_cv)
                    return error
                else:
                    pred=self.values(training=False,display_cv=True,display_task=True)
                    return [(np.hstack(pred[cv])-self.validation_data[cv].iloc[:,1].values)
                            for cv in range(0,self.cv_number)]
        else:
            if display_task:
                errors_cv_t=self.errors(training=training,display_cv=True,display_task=True)
                errors=[]
                for task in range(self.task_number):
                    errors_t=errors_cv_t[0][task]
                    for cv in range(1,self.cv_number):
                        errors_t=np.hstack((errors_t,errors_cv_t[cv][task]))
                    errors.append(errors_t)
                return errors
            else:
                return np.hstack(self.errors(training=training,display_cv=True,display_task=False))
        
    def total_errors(self,training=True,display_cv=False,display_task=False):
        """
        Return the errors over whole cross validation.
        """
        if training:
            return compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task),
                                samples_number=self.samples_number)
        else:
            return compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task),
                                samples_number=self.validation_samples_number)
        
    def drop(self,index=[]):
        index+=self.drop_index
        return Results(self.current_path,self.property_name,index)
