from . import utils

from collections import Iterable
import re
import os
import sys
import string
import functools
import math
import json

import pandas as pd
import numpy as np





def evaluate_expression(expression,data):
    r"""
    Returns the values using ``expression`` to compute ``data``.
    
    Arguments:
        expression (string): 
            arithmetic expression. 
            The expression should be in the same form as the descriptor in file ``SISSO.out``,
            i.e., every operation should be enclosed in parenthesis, like (exp((A+B))/((C*D))^2).
        
        data (pandas.DataFrame): 
            the data you want to compute, with samples as index and features as columns.
            The features name should be correspondent to the operands in expression.
    
    Returns:
        pandas.Series: values computed using ``expression``
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
                    elif OPTR[-2]=='(':
                        OPND.append(operand)
                        OPTR.pop()
                        i+=1
                        continue
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



def compute_using_model_reg(path=None,result=None,training=True,data=None,task_idx=None,dimension_idx=None):
    r"""
    Uses SISSO model of specific ``task_idx`` and ``dimension_idx`` to predict property of ``data``.
    This function is a little bit hard to use, you can also see :meth:`Regression.predict`
    
    Arguments:
        path (string): 
            directory path of SISSO result which contains the model you want to use.
        
        result (Regression): 
            instance of Regression which contains the model you want to use.
        
        training (bool): 
            whether the task is training or predicting. Default is True.
        
        data (string or pandas.DataFrame): 
            the path of the data or the data you want to compute. 
            If it is :class:`string`, it will be recognized as path to the data and 
            use ``data=pd.read_csv(data,sep=r'\s+')`` to read the data, so remember to use space to seperate the data.
            Otherwise it should be :class:`pandas.DataFrame`.
        
        task_idx (integer): 
            specifies which task of model you want to use.
        
        dimension_idx (integer): 
            specifies which dimension of model you want to use.
    
    Returns:
        Values computed using given model. The index order is [task, dimension, sample], it may not include
        all the index here, depending on you input.
    
    .. note::
        * You should only specify one of ``path`` or ``result`` to determine what model you want to use.
        
        * You only need to pass value to ``task_idx`` and ``dimension_idx`` when you want to use specific ``data``.
        
        * If you don't pass value to ``data``, you will get predictions of ``train.dat`` if ``training`` is ``True``
          or predictions of ``validation.dat`` if ``training`` is ``False``. 
          Also, in this case, you don't need to pass task_idx and dimension_idx, because you already specify which
          task of model the sample should use, and it will return all dimension results.
        
        * If you pass value to ``data``, then you have to specify ``task_idx``, so it means that
          all samples should be computed using the same task. 
          If you don't pass value to ``dimension_idx``, it will return values computed by all the dimension of models,
          otherwise it will only return values computed using ``dimension_idx`` of model.
    """
    
    if path:
        result=Regression(path)
    pred=[]
    if data:
        if isinstance(data,str):
            data=pd.read_csv(os.path.join(data),sep=r'\s+')
        if dimension_idx:
            for i in range(dimension_idx):
                value+=result.coefficients[task_idx-1][dimension_idx-1][i]*evaluate_expression(result.descriptors[dimension_idx-1][i],data)
            value+=result.intercepts[task_idx-1][dimension_idx-1]
            pred=value
        else:
            for dimension in range(result.dimension):
                for i in range(dimension):
                    value+=result.coefficients[task_idx-1][dimension][i]*evaluate_expression(result.descriptors[dimension][i],data)
                value+=result.intercepts[task_idx-1][dimension]
                pred.append(value)
            pred=np.array(value)
    else:
        if training:
            for task in range(result.n_task):
                pred_t=[]
                for dimension in range(result.dimension):
                    value=0
                    for i in range(dimension+1):
                        value+=result.coefficients[task][dimension][i]*evaluate_expression(result.descriptors[dimension][i],result.data_task[task])
                    value+=result.intercepts[task][dimension]
                    pred_t.append(list(value))
                pred.append(np.array(pred_t))
        else:
            for task in range(result.n_task):
                pred_t=[]
                for dimension in range(result.dimension):
                    value=0
                    for i in range(dimension+1):
                        value+=result.coefficients[task][dimension][i]*evaluate_expression(result.descriptors[dimension][i],result.validation_data_task[task])
                    value+=result.intercepts[task][dimension]
                    pred_t.append(list(value))
                pred.append(np.array(pred_t))
    return pred



def predict_reg(data,descriptors,coefficients,intercepts,tasks=None,dimensions=None):
    r"""
    Returns the predictions.
    
    Arguments:
        data (string or pandas.DataFrame): 
            the path of the data or the data you want to compute. 
            If it is :class:`string`, it will be recognized as path to the data and 
            use ``data=pd.read_csv(data,sep=r'\s+')`` to read the data, so remember to use space to seperate the data.
            Otherwise it should be :class:`pandas.DataFrame`.
        
        descriptors (list): 
            the descriptors you want to use. The index order should be dimension of model then dimension of desciptor.
        
        coefficients (list): 
            the coefficients you want to use. The index order should be task, dimension of model then dimension of desciptor.
        
        intercepts (float or list): 
            the intercepts you want to use. The index order should be task then dimension of model.
        
        tasks (None or list): 
            specifies which sample should use which task of model to compute.
            It should be task=[ [task_index, [sample_indices] ] ].
            For example, [ [1, [1,3]], [2, [2,4,5]] ] means sample 1 and 3 will be computed using task 1 model,
            and sample 2, 4 and 5 will be computed using task 2 model.
            If it is None, then compute all samples with task 1 model.
        
        dimensions (None or list): 
            specifies which dimension of desctiptor will be used.
            For example, [2,5] means only compute using 2D and 5D models.
            If it is None, then compute all samples with all dimension models.
        
    Returns:
        Predictions using passed models.
        
    .. note::
        * ``intercepts`` should be correspondent to other arguments.
        
        * If you just want to use 1 model, then ``descriptors``, ``coefficients`` shoule be a list with string and float as item, 
          and ``intercepts`` should be :class:`float`. You don't need to set ``tasks`` and ``dimensions`` in this case.
        
        * If you want to use many models, ``coefficients`` should be a list and index is different task.
          Each item in the list should also be a list and index is dimension of the model.
          Then each item of this list is also a list and index is dimension of descriptor, which contains specific descriptor or coefficient. 
          So the index order should be task, dimension of model then dimension of desciptor.
          Index order of ``descriptors`` should be dimension of model then dimension of desciptor.
          Index order of ``intercepts`` should be task then dimension of model.
    """
    
    # if data is string, then read the data.
    if isinstance(data,str):
            data=pd.read_csv(os.path.join(data),sep=r'\s+')
    
    # if the model contains different task and dimension
    if (isinstance(intercepts,Iterable) and isinstance(intercepts[0],Iterable)):
        
        # if tasks == None, then data will be computed using the first task
        if tasks==None:
            tasks=[[1,list(range(len(data)))]]
        
        # if dimensions == None, then data will be computed using all dimension
        if dimensions==None:
            dimensions=list(range(1,len(intercepts[0])+1))
        
        # compute the number of samples
        n_sample=0
        for task in tasks:
            n_sample+=len(task[1])
        
        # compute the predictions
        pred=np.zeros((len(dimensions),n_sample))
        for task in tasks:
            task_idx=task[0]-1
            sample_idxs=task[1]
            d=0
            for dimension in dimensions:
                for i in range(dimension):
                    pred[d][sample_idxs]+=coefficients[task_idx][dimension-1][i]*evaluate_expression(descriptors[dimension-1][i],data.iloc[sample_idxs])
                pred[d][sample_idxs]+=intercepts[task_idx][dimension-1]
                d+=1
    
    # only 1 model
    elif isinstance(intercepts,float):
        pred=0
        for i in range(len(coefficients)):
            pred+=coefficients[i]*evaluate_expression(descriptors[i],data)
        pred+=intercepts
    
    return pred



def compute_errors(errors):
    r"""    
    Computes errors.
    
    Arguments:
        errors:
            difference between predictions and exact value.
        
    Returns:
        RMSE, MAE, 25%ile AE, 50%ile AE, 75%ile AE, 95%ile AE, MaxAE of given errors.
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
        n_task=len(errors)
        n_dimension=len(errors[0])
        error=[]
        for task in range(n_task):
            error.append(pd.DataFrame([compute_errors(errors[task][dimension]) for dimension in range(n_dimension)],
                                    columns=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],
                                    index=list(range(1,n_dimension+1))))
    
    # 4D [cv, task, dimension, sample]
    elif isinstance(errors,list) and isinstance(errors[0],list):
        cv_num=len(errors)
        n_task=len(errors[0])
        n_dimension=len(errors[0][0])
        error=[]
        for cv in range(cv_num):
            error_cv=[]
            for task in range(n_task):
                error_cv.append(pd.DataFrame([compute_errors(errors[cv][task][dimension]) for dimension in range(n_dimension)],
                                            columns=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],
                                            index=list(range(1,n_dimension+1))))
            error.append(error_cv)
    
    # 2D [dimension, sample]
    elif errors.ndim==2:
        n_dimension=len(errors)
        error=pd.DataFrame([compute_errors(errors[dimension]) for dimension in range(n_dimension)],
                            columns=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],
                            index=list(range(1,n_dimension+1)))
        
    return error





class Regression(object):
    r"""
    Basic class for evaluating the results of regression. You should instantiate it first.
    
    Arguments:
        current_path (string): 
            path to the directory of the SISSO result.
    
    Its attributes contain all the input arguments in ``SISSO.in``, information about ``train.dat``
    and ``SISSO.out``.
    """
    
    def __init__(self,current_path):
        self.current_path=current_path
        
        # ------------------- read arguments of SISSO.in ----------------------------
        
        with open(os.path.join(self.current_path,'SISSO.in'),'r') as f:
            input_file=f.read()
            
            # keywords for the target properties
            n_task=int(re.findall(r'ntask\s*=\s*(\d+)',input_file)[0])
            n_sample=re.findall(r'nsample\s*=\s*([\d, ]+)',input_file)[0]
            n_sample=re.split(r'[,\s]+',n_sample)
            n_sample=list(filter(lambda x: bool(re.search(r'\d',x)),n_sample))
            #if n_sample[-1]=='':
            #    n_sample=n_sample[:-1]
            n_sample=list(map(int,n_sample))
            task_weighting=int(re.findall(r'task_weighting\s*=\s*(\d+)',input_file)[0])
            dimension=int(re.findall(r'desc_dim\s*=\s*(\d+)',input_file)[0])
            
            # keywords for feature construction and sure independence screening
            rung=int(re.findall(r'rung\s*=\s*(\d+)',input_file)[0])
            operation_set=re.findall(r"opset\s*=\s*'(.+)'",input_file)
            operation_set=re.split(r'[\(\)]+',operation_set[0])[1:-1]
            maxcomplexity=int(re.findall(r'maxcomplexity\s*=\s*(\d+)',input_file)[0])
            dimclass=re.findall(r'dimclass=([\(\d:\)]+)',input_file)[0]
            maxfval_lb=float(re.findall(r'maxfval_lb\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            maxfval_ub=float(re.findall(r'maxfval_ub\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            subs_sis=int(re.findall(r'subs_sis\s*=\s*(\d+)',input_file)[0])
            
            # keywords for descriptor
            method=re.findall(r'method\s*=\s*\'(\w+)',input_file)[0]
            L1L0_size4LO=int(re.findall(r'L1L0_size4L0\s*=\s*(\d+)',input_file)[0])
            fit_intercept=re.findall(r'fit_intercept\s*=\s*.(\w+).',input_file)[0]
            if fit_intercept=='true':
                fit_intercept=True
            else:
                fit_intercept=False
            metric=re.findall(r'metric\s*=\s*\'(\w+)',input_file)[0]
            nm_output=int(re.findall(r'nm_output\s*=\s*(\d+)',input_file)[0])
        
        # keywords for the target properties
        self.n_task=n_task
        self.n_sample=n_sample
        self.task_weighting=task_weighting
        self.dimension=dimension
        
        # keywords for feature construction and sure independence screening
        self.rung=rung
        self.operation_set=operation_set
        self.maxcomplexity=maxcomplexity
        self.dimclass=dimclass
        self.maxfval_lb=maxfval_lb
        self.maxfval_ub=maxfval_ub
        self.subs_sis=subs_sis
        
        # keywords for descriptor
        self.method=method
        self.L1L0_size4LO=L1L0_size4LO
        self.fit_intercept=fit_intercept
        self.metric=metric
        self.nm_output=nm_output
        
        # ------------------------------ read data --------------------------------
        
        self.data=pd.read_csv(os.path.join(current_path,'train.dat'),sep=r'\s+')
        self.materials=self.data.iloc[:,0]
        self.property=self.data.iloc[:,1]
        self.property_name=self.data.columns.tolist()[1]
        self.features_name=self.data.columns.tolist()[2:]
        
        
        # -------------------------- read data per task ----------------------------
        
        self.data_task=utils.seperate_DataFrame(self.data,self.n_sample)
        self.materials_task=utils.seperate_DataFrame(self.materials,self.n_sample)
        self.property_task=utils.seperate_DataFrame(self.property,self.n_sample)
        
        # ------------------------- read validation data ---------------------------
        
        if os.path.exists(os.path.join(self.current_path,'validation.dat')):
            self.validation_data=pd.read_csv(os.path.join(current_path,'validation.dat'),sep=r'\s+')
            with open(os.path.join(self.current_path,'shuffle.dat'),'r') as f:
                shuffle=json.load(f)
            self.n_sample=shuffle['training_samples_number']
            self.n_validation_sample=shuffle['validation_samples_number']
            self.validation_data_task=utils.seperate_DataFrame(self.validation_data,self.n_validation_sample)

    def __repr__(self):
        text='#'*50+'\n'+'SISSO Regression\n'+'#'*50
        text+='\nProperty Name: %s\nSample Number: %d\nTask Number: %d\nRung: %d\nDimension: %d\nSubs_sis: %d\n'%(self.property_name,self.n_sample,self.n_task,self.rung,self.dimension,self.subs_sis)
        return text
    
    @property
    def baseline(self):
        r"""
        Returns the baseline, i.e., the errors with predicting every property using the mean value of the property in ``train.dat``.
        """
        
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
    
    def get_descriptors(self,path=None):
        r"""
        Returns descriptors.
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
    
    @property
    def descriptors(self):
        r"""
        Returns descriptors.
        """
        
        return self.get_descriptors()
    
    def get_coefficients(self,path=None):
        r"""
        Returns coefficients. The index order is task, dimension of model, dimension descriptor.
        """
        
        if path==None:
            path=self.current_path
        coefficients_all=[]
        for task in range(0,self.n_task):
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
    
    @property
    def coefficients(self):
        r"""
        Returns coefficients. The index order is task, dimension of model, dimension descriptor.
        """
        
        return self.get_coefficients()
    
    def get_intercepts(self,path=None):
        r"""
        Returns intercepts. The index order is task, dimension of model.
        """
        
        if path==None:
            path=self.current_path
        intercepts_all=[]
        for task in range(0,self.n_task):
            with open(os.path.join(path,'SISSO.out'),'r') as f:
                input_file=f.read()
                intercepts_t=re.findall(r'Intercept_00%d:(.*)'%(task+1),input_file)
                intercepts_t=list(map(float,intercepts_t))
            intercepts_all.append(intercepts_t)
        return intercepts_all
    
    @property
    def intercepts(self):
        r"""
        Returns intercepts. The index order is task, dimension of model.
        """
        
        return self.get_intercepts()
    
    def features_percent(self,descending=True):
        r"""
        Computes the percentages of each feature in top subs_sis 1D descriptors.
        """
        
        feature_space=pd.read_csv(os.path.join(self.current_path,'feature_space','Uspace.name'),sep=r'\s+',header=None).iloc[0:self.subs_sis,0]
        feature_percent=pd.DataFrame(columns=self.features_name,index=('percent',))
        for feature_name in self.features_name:
            percent=feature_space.str.contains(feature_name).sum()/self.subs_sis
            feature_percent.loc['percent',feature_name]=percent
        feature_percent=feature_percent.T
        if descending:
            feature_percent.sort_values('percent',ascending=False,inplace=True)
        return feature_percent
    
    def evaluate_expression(self,expression,data=None):
        r"""
        Returns the value computed using given expression over data in ``train.dat``.
        
        Returns the values using ``expression`` to compute ``data``.
    
        Arguments:
            expression (string): 
            arithmetic expression. 
                The expression should be in the same form as the descriptor in file ``SISSO.out``,
                i.e., every operation should be enclosed in parenthesis, like (exp((A+B))/((C*D))^2).
            
            data (pandas.DataFrame): 
                the data you want to compute, with samples as index and features as columns.
                The features name should be correspondent to the operands in expression.
        
        Returns:
            pandas.Series: values computed using ``expression``
        """
        
        if isinstance(data,str):
            data=pd.read_csv(os.path.join(data),sep=r'\s+')
        if data == None:
            data=self.data
        return evaluate_expression(expression,data)
    
    def predict(self,data,tasks=None,dimensions=None):
        r"""
        Returns the predictions of ``data`` using the models found by SISSO.
        
        Arguments:
            data (string or pandas.DataFrame): 
                the path of the data or the data you want to compute. 
                If it is :class:`string`, it will be recognized as path to the data and 
                use ``data=pd.read_csv(data,sep=r'\s+')`` to read the data, so remember to use space to seperate the data.
                Otherwise it should be :class:`pandas.DataFrame`.
            
            tasks (None or list):  
                specifies which sample should use which task of model to compute.
                It should be task=[ [task_index, [sample_indices] ] ].
                For example, [ [1, [1,3]], [2, [2,4,5]] ] means sample 1 and 3 will be computed using task 1 model,
                and sample 2, 4 and 5 will be computed using task 2 model.
                If it is None, then compute all samples with task 1 model.
            
            dimensions (None or list):  
                specifies which dimension of desctiptor will be used.
                For example, [2,5] means only compute using 2D and 5D models.
                If it is None, then compute all samples with all dimension models.
        
        Returns:
            Values computed using models found by SISSO.
        """
        
        if isinstance(data,str):
            data=pd.read_csv(os.path.join(data),sep=r'\s+')
        return predict_reg(data,self.descriptors,self.coefficients,self.intercepts,tasks=tasks,dimensions=dimensions)
    
    def predictions(self,training=True,display_task=False):
        r"""
        Returns predictions.
        The index order is dimension, sample if ``display_task`` is ``False``,
        or task, sample, dimension if ``display_task`` is ``True``.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the predictions contain index of task or not.
        
        Returns:
            Predictions.
        """
        
        if display_task==True:
            return compute_using_model_reg(result=self,training=training)
        else:
            return np.hstack(compute_using_model_reg(result=self,training=training))
    
    def errors(self,training=True,display_task=False):
        r"""
        Returns errors.
            The index order is dimension, sample if ``display_task`` is ``False``,
            or task, sample, dimension if ``display_task`` is ``True``.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the errors contain index of task or not.
        
        Returns:
            Errors.
        """
        
        if display_task==True:
            pred=self.predictions(training=training,display_task=True)
            return [pred[task]-self.property_task[task].values for task in range(0,self.n_task)]
        else:
            if training==True:
                return self.predictions(training=training,display_task=False)-self.property.values
            else:
                return self.predictions(training=training,display_task=False)-self.validation_data.iloc[:,1].values
    
    def total_errors(self,training=True,display_task=False,display_baseline=False):
        r"""    
        Computes errors.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the errors contain index of task or not.
            
            display_baseline (bool):  
                determines whether display baseline
            
        Returns:
            RMSE, MAE, 25%ile AE, 50%ile AE, 75%ile AE, 95%ile AE, MaxAE of given errors.
        """
        
        if display_baseline:
            return pd.concat([pd.DataFrame(self.baseline[1:].values,
                                        columns=['Baseline'],
                                        index=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE']).T,
                            compute_errors(self.errors(training=training,display_task=display_task))])
        else:
            return compute_errors(self.errors(training=training,display_task=display_task))
        
    def check_predictions(self,dimension,multiply_coefficients=True):
        r"""
        Checks predictions of each descriptor.
        
        Arguments:  
            dimension (int):
                dimension of the descriptor.
                
            multiply_coefficients (bool):  
                whether it should be multiplied by coefficients.
            
        Returns:
            Predictions of each descriptor.
        """
        
        descriptors=self.descriptors
        coefficients=self.coefficients
        n_sample=self.n_sample
        data=self.data_task
        predictions=np.zeros([dimension+1,np.array(n_sample).sum()])
        total_n_sample=0
        for task in range(len(n_sample)):
            predictions[0,total_n_sample:total_n_sample+n_sample[task]]=self.intercepts[task][dimension-1]
            total_n_sample+=n_sample[task]
        if multiply_coefficients:
            for d in range(dimension):
                total_n_sample=0
                for task in range(len(n_sample)):
                    predictions[d+1,total_n_sample:total_n_sample+n_sample[task]]=coefficients[task][dimension-1][d]*evaluate_expression(descriptors[dimension-1][d],data[task]).values
                    total_n_sample+=n_sample[task]
        else:
            for d in range(dimension):
                total_n_sample=0
                for task in range(len(n_sample)):
                    predictions[d+1,total_n_sample:total_n_sample+n_sample[task]]=evaluate_expression(descriptors[dimension-1][d],data[task]).values
                    total_n_sample+=n_sample[task]
        return predictions
    
    def check_percentage(self,dimension,absolute=True):
        r"""
        Checks percentage of each descriptors.
        
        Arguments:  
            dimension (int):
                dimension of the descriptor.
                
            absolute (bool):
                whether return the absolute descriptor.
                If it is ``True``, the numerator is absolute value of each descriptor multiplied by corresponding coefficient,
                and denominator is the sum of intercept and every descriptor multiplied by coefficient.
                If it is ``False``, the numerator is descriptor multiplied by corresponding coefficient,
                and denominator is the proeprty. In this case, sometimes it will be larger than 1.
                
        Returns:
            percentage which index is [dimension, sample].
        """
        
        predictions=self.check_predictions(dimension)
        if absolute:
            return np.abs(predictions)/np.sum(np.abs(predictions),axis=0)
        else:
            return predictions/self.property.values





class RegressionCV(Regression):
    r"""
    Basic class for evaluating the cross validation results of regression.
    You should instantiate it first if you want to analyze CV results.
    You can use index to select a specific result from total CV results, and it will return
    a instance of :class:`Regression`.
    
    Arguments:
        current_path (string):  
            path to the directory of the cross validation results.
        
        property_name (string):  
            specifies the property name of your CV results.
        
        drop_index (list):  
            specifies which CV results you don't want to consider.
    
    .. note::
        You should use code in ``cross_validation.py`` to generate the CV files, otherwise the format
        may be wrong. 
        
        Its attributes contain all the input arguments in ``SISSO.in``, information about ``train.dat``,
        ``validation.dat``, and ``SISSO.out``.
    """
    
    def __init__(self,current_path,property_name=None,drop_index=[]):
        # -------------------------- read path and CV info----------------------------
        cv_names=sorted(list(filter(lambda x: '_cv' in x,os.listdir(current_path))),
                        key=lambda x:int(x.split('_cv')[-1]))
        dir_list=list(map(lambda cv_name:os.path.join(current_path,cv_name),cv_names))
        n_cv=len(cv_names)
        
        self.current_path=current_path
        self.cv_path=[dir_list[cv] for cv in range(n_cv) if cv not in drop_index]
        self.drop_index=drop_index
        self.n_cv=n_cv-len(drop_index)
        
        with open(os.path.join(self.current_path,'cross_validation_info.dat'),'r') as f:
            cv_info=json.load(f)
        self.cross_validation_type=cv_info['cross_validation_type']
        
        # ------------------- read arguments of SISSO.in ----------------------------
        with open(os.path.join(self.cv_path[0],'SISSO.in'),'r') as f:
            input_file=f.read()
            
            # keywords for the target properties
            n_task=int(re.findall(r'ntask\s*=\s*(\d+)',input_file)[0])
            task_weighting=int(re.findall(r'task_weighting\s*=\s*(\d+)',input_file)[0])
            dimension=int(re.findall(r'desc_dim\s*=\s*(\d+)',input_file)[0])
            
            # keywords for feature construction and sure independence screening
            rung=int(re.findall(r'rung\s*=\s*(\d+)',input_file)[0])
            operation_set=re.findall(r"opset\s*=\s*'(.+)'",input_file)
            operation_set=re.split(r'[\(\)]+',operation_set[0])[1:-1]
            maxcomplexity=int(re.findall(r'maxcomplexity\s*=\s*(\d+)',input_file)[0])
            dimclass=re.findall(r'dimclass=([\(\d:\)]+)',input_file)[0]
            maxfval_lb=float(re.findall(r'maxfval_lb\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            maxfval_ub=float(re.findall(r'maxfval_ub\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            subs_sis=int(re.findall(r'subs_sis\s*=\s*(\d+)',input_file)[0])
            
            # keywords for descriptor
            method=re.findall(r'method\s*=\s*\'(\w+)',input_file)[0]
            L1L0_size4LO=int(re.findall(r'L1L0_size4L0\s*=\s*(\d+)',input_file)[0])
            fit_intercept=re.findall(r'fit_intercept\s*=\s*.(\w+).',input_file)[0]
            if fit_intercept=='true':
                fit_intercept=True
            else:
                fit_intercept=False
            metric=re.findall(r'metric\s*=\s*\'(\w+)',input_file)[0]
            nm_output=int(re.findall(r'nm_output\s*=\s*(\d+)',input_file)[0])
        
        # keywords for the target properties
        self.n_task=n_task
        self.task_weighting=task_weighting
        self.dimension=dimension
        
        # keywords for feature construction and sure independence screening
        self.rung=rung
        self.operation_set=operation_set
        self.maxcomplexity=maxcomplexity
        self.dimclass=dimclass
        self.maxfval_lb=maxfval_lb
        self.maxfval_ub=maxfval_ub
        self.subs_sis=subs_sis
        
        # keywords for descriptor
        self.method=method
        self.L1L0_size4LO=L1L0_size4LO
        self.fit_intercept=fit_intercept
        self.metric=metric
        self.nm_output=nm_output
        
        # --------------------------- read total data -----------------------------
        
        self.total_data=pd.read_csv(os.path.join(current_path,'train.dat'),sep=r'\s+')
        self.n_total_sample=len(pd.read_csv(os.path.join(current_path,'train.dat'),sep=r'\s+'))
        self.property_name=property_name if property_name else cv_names[0].split('_cv')[0]
        
        # -------------------------- read data per CV -----------------------------
        
        self.data=[pd.read_csv(os.path.join(cv_path,'train.dat'),sep=r'\s+') for cv_path in self.cv_path]
        self.materials=[data.iloc[:,0] for data in self.data]
        self.property=[data.iloc[:,1] for data in self.data]
        self.features_name=self.data[0].columns.tolist()[2:]
        self.n_sample=np.array([json.load(open(os.path.join(cv_path,'shuffle.dat'),'r'))['training_samples_number']
                        for cv_path in self.cv_path])
        self.n_validation_sample=np.array([json.load(open(os.path.join(cv_path,'shuffle.dat'),'r'))['validation_samples_number']
                        for cv_path in self.cv_path])
        self.validation_data=[pd.read_csv(os.path.join(cv_path,'validation.dat'),sep=r'\s+') for cv_path in self.cv_path]
        
        # --------------------- read data per CV per task ------------------------
        
        self.data_task=[utils.seperate_DataFrame(self.data[cv],self.n_sample[cv])
                        for cv in range(0,self.n_cv)]
        self.materials_task=[utils.seperate_DataFrame(self.materials[cv],self.n_sample[cv])
                        for cv in range(0,self.n_cv)]
        self.property_task=[utils.seperate_DataFrame(self.property[cv],self.n_sample[cv])
                        for cv in range(0,self.n_cv)]
        self.validation_data_task=[utils.seperate_DataFrame(self.validation_data[cv],self.n_validation_sample[cv])
                        for cv in range(0,self.n_cv)]
    
    def __getitem__(self,index):
        if isinstance(index,slice):
            return [Regression(self.cv_path[i]) for i in range(index.start,index.stop)]
        else:
            return Regression(self.cv_path[index])
    
    def __repr__(self):
        text='#'*50+'\n'+'SISSO Regression CV\n'+'#'*50
        with open(os.path.join(self.current_path,'cross_validation_info.dat'),'r') as f:
            cv_info=json.load(f)
        if 'shuffle_data_list' in cv_info:
            text+=('\nCross Validation Type: %s\nShuffle Data List: '%cv_info['cross_validation_type']+str(cv_info['shuffle_data_list']))
        else:
            text+=('\nCross Validation Type: %s\nIteration: '%cv_info['cross_validation_type']+str(self.n_cv))
        text+='\nProperty Name: %s\nTask Number: %d\nRung: %d\nDimension: %d\nSubs_sis: %d'%(self.property_name,self.n_task,self.rung,self.dimension,self.subs_sis)
        return text
    
    @property
    def baseline(self):
        r"""
        Returns the baseline, i.e., the errors with predicting every property using the mean value of the property in ``train.dat``.
        """
        
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
        r"""
        Returns the samples' names and in which CV result they are.
        
        Arguments:
            idx (integer):  
                index of the sample.
        
        Returns:
            the samples' names and in which CV result they are.
        """
        
        if self.cross_validation_type.startswith('leave'):
            val_num=len(self.validation_data[0])
            return ([self.validation_data[int(index/val_num)].iloc[index%val_num,0] for index in idxs],
                    [int(self.cv_path[int(index/val_num)].split('cv')[-1]) for index in idxs])
    
    def find_max_error(self):
        r"""
        Returns the samples' names and in which CV result they are that contribute to the maxAE.
        """
        
        return self.find_materials_in_validation(*(np.argmax(np.abs(self.errors(training=False)),axis=1)))
    
    @property
    def descriptors(self):
        r"""
        Returns descriptors. The first index is cross validation index.
        """
        
        return [super(RegressionCV,self).get_descriptors(path=cv_path) for cv_path in self.cv_path]
    
    @property
    def coefficients(self):
        r"""
        Returns coefficients. The index order is CV index, task, dimension of model, dimension descriptor.
        """
        
        return [super(RegressionCV,self).get_coefficients(path=cv_path) for cv_path in self.cv_path]
    
    @property
    def intercepts(self):
        r"""
        Returns intercepts. The index order is CV index, task, dimension of model.
        """
        
        return [super(RegressionCV,self).get_intercepts(path=cv_path) for cv_path in self.cv_path]
    
    def features_percent(self,descending=True):
        r"""
        Return the percent of each feature in the top subs_sis descriptors.
        There are total n_cv*subs_sis descriptors,
        the feature percent is the percent over these descriptors.
        """
        
        feature_percent=pd.DataFrame(columns=self.features_name,index=('percent',))
        feature_percent.iloc[0,:]=0
        for cv_path in self.cv_path:
            feature_space=pd.read_csv(os.path.join(cv_path,'feature_space','Uspace.name'),sep=r'\s+',header=None).iloc[0:self.subs_sis,0]
            for feature_name in self.features_name:
                count=feature_space.str.contains(feature_name).sum()
                feature_percent.loc['percent',feature_name]+=count
        feature_percent.iloc[0,:]=feature_percent.iloc[0,:]/(self.n_cv*self.subs_sis)
        feature_percent=feature_percent.T
        if descending:
            feature_percent.sort_values('percent',ascending=False,inplace=True)
        return feature_percent
    
    def descriptor_percent(self,descriptor):
        r"""
        Return the percent of given descriptor appearing in the cross validation top subs_sis descriptors,
        and return the appearing index in the descriptor space.
        
        Arguments:
            descriptor (string):  
                the descriptor you want to check.
        """
        
        count=0
        descriptor_index=np.zeros(self.n_cv)
        for cv in range(0,self.n_cv):
            feature_space=pd.read_csv(os.path.join(self.cv_path[cv],'feature_space','Uspace.name'),sep=r'\s+',header=None).iloc[0:self.subs_sis,0]
            try:
                descriptor_index[cv]=feature_space.tolist().index(descriptor)+1
                count+=1
            except ValueError:
                descriptor_index[cv]=None
        return count/self.n_cv,descriptor_index
    
    def predict(self,data,cv_index=None,tasks=None,dimensions=None):
        r"""
        Returns the predictions of ``data`` using the models found by SISSO.
        
        Arguments:
            data (string or pandas.DataFrame):  
                the path of the data or the data you want to compute. 
                If it is :class:`string`, it will be recognized as path to the data and 
                use ``data=pd.read_csv(data,sep=r'\s+')`` to read the data, so remember to use space to seperate the data.
                Otherwise it should be :class:`pandas.DataFrame`.
            
            cv_index (None or list): 
                specifies which CV should be included.
                For example, [1,5] means CV1 and CV5 will be included.
                If it is None, then compute all CV results.
            
            tasks (None or list):  
                specifies which sample should use which task of model to compute.
                It should be task=[ [task_index, [sample_indices] ] ].
                For example, [ [1, [1,3]], [2, [2,4,5]] ] means sample 1 and 3 will be computed using task 1 model,
                and sample 2, 4 and 5 will be computed using task 2 model.
                If it is None, then compute all samples with task 1 model.
            
            dimensions (None or list):  
                specifies which dimension of desctiptor will be used.
                For example, [2,5] means only compute using 2D and 5D models.
                If it is None, then compute all samples with all dimension models.
        
        Returns:
            Values computed using models found by SISSO.
        """
        
        if cv_index==None:
            cv_index=list(range(self.n_cv))
        
        return [predict_reg(data,self.descriptors[cv],self.coefficients[cv],self.intercepts[cv],tasks=tasks,dimensions=dimensions)
                for cv in cv_index]
    
    def predictions(self,training=True,display_cv=False,display_task=False):
        r"""
        Returns predictions.
        The index order is dimension, sample if ``display_cv`` is ``False`` and ``display_task`` is ``False``,
        or task, sample, dimension if ``display_cv`` is ``False`` and ``display_task`` is ``True``,
        or CV index, sample, dimension if ``display_cv`` is ``True`` and ``display_task`` is ``False``,
        or CV index, task, sample, dimension if ``display_cv`` is ``True`` and ``display_task`` is ``True``.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the predictions contain index of task or not.
        
        Returns:
            Predictions.
        """
        
        if display_cv==True:
            if display_task==True:
                return [compute_using_model_reg(path=cv_path,
                                        training=training)
                for cv_path in self.cv_path]
            else:
                return [np.hstack(compute_using_model_reg(path=cv_path,
                                        training=training))
                for cv_path in self.cv_path]
        else:
            return np.hstack(self.predictions(training=training,display_cv=True,display_task=False))
    
    def errors(self,training=True,display_cv=False,display_task=False):
        r"""
        Returns errors.
        The index order is dimension, sample if ``display_cv`` is ``False`` and ``display_task`` is ``False``,
        or task, sample, dimension if ``display_cv`` is ``False`` and ``display_task`` is ``True``,
        or CV index, sample, dimension if ``display_cv`` is ``True`` and ``display_task`` is ``False``,
        or CV index, task, sample, dimension if ``display_cv`` is ``True`` and ``display_task`` is ``True``.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the errors contain index of task or not.
        
        Returns:
            Errors.
        """
        
        if display_cv:
            if training:
                if display_task:
                    error=[]
                    pred=self.predictions(training=True,display_cv=True,display_task=True)
                    for cv in range(self.n_cv):
                        error_cv=[]
                        for task in range(0,self.n_task):
                            error_cv.append(pred[cv][task]-self.property_task[cv][task])
                        error.append(error_cv)
                    return error
                else:
                    pred=self.predictions(training=True,display_cv=True,display_task=True)
                    return [np.hstack(pred[cv])-np.hstack(self.property_task[cv])
                            for cv in range(self.n_cv)]
            else:
                if display_task:
                    error=[]
                    pred=self.predictions(training=False,display_cv=True,display_task=True)
                    for cv in range(0,self.n_cv):
                        error_cv=[]
                        for task in range(self.n_task):
                            error_cv.append(pred[cv][task]-self.validation_data_task[cv][task].iloc[:,1].values)
                        error.append(error_cv)
                    return error
                else:
                    pred=self.predictions(training=False,display_cv=True,display_task=True)
                    return [(np.hstack(pred[cv])-self.validation_data[cv].iloc[:,1].values)
                            for cv in range(self.n_cv)]
        else:
            if display_task:
                errors_cv_t=self.errors(training=training,display_cv=True,display_task=True)
                errors=[]
                for task in range(self.n_task):
                    errors_t=errors_cv_t[0][task]
                    for cv in range(1,self.n_cv):
                        errors_t=np.hstack((errors_t,errors_cv_t[cv][task]))
                    errors.append(errors_t)
                return errors
            else:
                return np.hstack(self.errors(training=training,display_cv=True,display_task=False))
        
    def total_errors(self,training=True,display_cv=False,display_task=False,display_baseline=False):
        r"""    
        Compute errors.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_cv (bool):  
                determines the errors contain index of CV or not.
            
            display_task (bool):  
                determines the errors contain index of task or not.
            
            display_baseline (bool):  
                determines whether display baseline
            
        Returns:
            RMSE, MAE, 25%ile AE, 50%ile AE, 75%ile AE, 95%ile AE, MaxAE of given errors.
        """
        
        if display_baseline:
            if training:
                return pd.concat([pd.DataFrame(self.baseline[1:].values,
                                        columns=['Baseline'],
                                        index=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE']).T,
                                compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task))])
            else:
                return pd.concat([pd.DataFrame(self.baseline[1:].values,
                                        columns=['Baseline'],
                                        index=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE']).T,
                                compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task))])
        else:
            if training:
                return compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task))
            else:
                return compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task))
        
    def drop(self,index=[]):
        r"""
        Drop some CV results.
        
        Arguments:
            index (list):  
                contain the CV index which you want to drop.
        
        Returns:
            An instance of RegressionCV that drop the results in CV index.
        """
        
        index+=self.drop_index
        return RegressionCV(self.current_path,self.property_name,index)
    
    def check_predictions(self,cv_idx,dimension,training=True,multiply_coefficients=True):
        r"""
        Check predictions of each descriptor.
        
        Arguments:
            cv_idx (integer):  
                specifies which CV result you want to check.
                
            dimension (int):
                dimension of the descriptor.
            
            training (bool):  
                whether it is training.
            
            multiply_coefficients (bool):  
                whether it should be multiplied by coefficients.
            
        Returns
            Predictions of each descriptor.
        """
        
        descriptors=self.descriptors[cv_idx]
        coefficients=self.coefficients[cv_idx]
        if training:
            n_sample=self.n_sample[cv_idx]
            data=self.data_task[cv_idx]
        else:
            n_sample=self.n_validation_sample[cv_idx]
            data=self.validation_data_task[cv_idx]
        predictions=np.zeros([dimension+1,n_sample.sum()])
        total_n_sample=0
        for task in range(len(n_sample)):
            predictions[0,total_n_sample:total_n_sample+n_sample[task]]=self.intercepts[cv_idx][task][dimension-1]
            total_n_sample+=n_sample[task]
        if multiply_coefficients:
            for d in range(dimension):
                total_n_sample=0
                for task in range(len(n_sample)):
                    predictions[d+1,total_n_sample:total_n_sample+n_sample[task]]=coefficients[task][dimension-1][d]*evaluate_expression(descriptors[dimension-1][d],data[task]).values
                    total_n_sample+=n_sample[task]
        else:
            for d in range(dimension):
                total_n_sample=0
                for task in range(len(n_sample)):
                    predictions[d+1,total_n_sample:total_n_sample+n_sample[task]]=evaluate_expression(descriptors[dimension-1][d],data[task]).values
                    total_n_sample+=n_sample[task]
        return predictions
    
    def check_percentage(self,cv_idx,dimension,absolute=True):
        r"""
        Checks percentage of each descriptors.
        
        Arguments:  
            dimension (int):
                dimension of the descriptor.
                
            absolute (bool):
                whether return the absolute descriptor.
                If it is ``True``, the numerator is absolute value of each descriptor multiplied by corresponding coefficient,
                and denominator is the sum of intercept and every descriptor multiplied by coefficient.
                If it is ``False``, the numerator is descriptor multiplied by corresponding coefficient,
                and denominator is the proeprty. In this case, sometimes it will be larger than 1.
                
        Returns:
            percentage which index is [dimension, sample].
        """
        
        predictions=self.check_predictions(cv_idx,dimension)
        if absolute:
            return np.abs(predictions)/np.sum(np.abs(predictions),axis=0)
        else:
            return predictions/self.property[cv_idx].values





class Classification(object):
    r"""
    Basic class for evaluating the results of classification. You should instantiate it first.
    
    Arguments:
        current_path (string):  
            path to the directory of the SISSO result.
    
    Its attributes contain all the input arguments in ``SISSO.in``, information about ``train.dat``
    and ``SISSO.out``.
    """
    
    def __init__(self,current_path):
        self.current_path=current_path
        
        # ------------------- read arguments of SISSO.in ----------------------------
        
        with open(os.path.join(self.current_path,'SISSO.in'),'r') as f:
            input_file=f.read()
            
            # keywords for the target properties
            n_task=int(re.findall(r'ntask\s*=\s*(\d+)',input_file)[0])
            n_sample=re.findall(r'nsample\s*=\s*([\(\)\d,\s]*)\)',input_file)[0]
            n_sample=re.split(r'\(|\s',n_sample)
            n_sample=list(filter(lambda x: bool(re.search(r'\d',x)),n_sample))
            n_sample=[list(map(lambda d: int(d),filter(lambda x: bool(re.search(r'\d',x)),re.split(r',|\s|\)',n)))) for n in n_sample]   
            dimension=int(re.findall(r'desc_dim\s*=\s*(\d+)',input_file)[0])
            
            # keywords for feature construction and sure independence screening
            rung=int(re.findall(r'rung\s*=\s*(\d+)',input_file)[0])
            operation_set=re.findall(r"opset\s*=\s*'(.+)'",input_file)
            operation_set=re.split(r'[\(\)]+',operation_set[0])[1:-1]
            maxcomplexity=int(re.findall(r'maxcomplexity\s*=\s*(\d+)',input_file)[0])
            dimclass=re.findall(r'dimclass=([\(\d:\)]+)',input_file)[0]
            maxfval_lb=float(re.findall(r'maxfval_lb\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            maxfval_ub=float(re.findall(r'maxfval_ub\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            subs_sis=int(re.findall(r'subs_sis\s*=\s*(\d+)',input_file)[0])
            
            # keywords for descriptor
            method=re.findall(r'method\s*=\s*\'(\w+)',input_file)[0]
            isconvex=re.findall(r'isconvex\s*=\s*([\(\)\d,\s]*)\)',input_file)[0]
            isconvex=re.split(r'\(|\s',isconvex)
            isconvex=list(filter(lambda x: bool(re.search(r'\d',x)),isconvex))
            isconvex=[list(map(lambda d: int(d),filter(lambda x: bool(re.search(r'\d',x)),re.split(r',|\s|\)',n)))) for n in isconvex]   
            width=float(re.findall(r'width\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            nm_output=int(re.findall(r'nm_output\s*=\s*(\d+)',input_file)[0])
        
        # keywords for the target properties
        self.n_task=n_task
        self.n_sample=n_sample
        self.dimension=dimension
        
        # keywords for feature construction and sure independence screening
        self.rung=rung
        self.operation_set=operation_set
        self.maxcomplexity=maxcomplexity
        self.dimclass=dimclass
        self.maxfval_lb=maxfval_lb
        self.maxfval_ub=maxfval_ub
        self.subs_sis=subs_sis
        
        # keywords for descriptor
        self.method=method
        self.isconvex=isconvex
        self.width=width
        self.nm_output=nm_output
        
        # ------------------------------ read data --------------------------------
        
        self.data=pd.read_csv(os.path.join(current_path,'train.dat'),sep=r'\s+')
        self.materials=self.data.iloc[:,0]
        self.features_name=self.data.columns.tolist()[1:]
        
        
        # -------------------------- read data per task ----------------------------
        
        self.data_sep=utils.seperate_DataFrame(self.data,self.n_sample)
        self.materials_sep=utils.seperate_DataFrame(self.materials,self.n_sample)
        
        # ------------------------- read validation data ---------------------------
        
        if os.path.exists(os.path.join(self.current_path,'validation.dat')):
            self.validation_data=pd.read_csv(os.path.join(current_path,'validation.dat'),sep=r'\s+')
            with open(os.path.join(self.current_path,'shuffle.dat'),'r') as f:
                shuffle=json.load(f)
            self.n_sample=shuffle['training_samples_number']
            self.n_validation_sample=shuffle['validation_samples_number']
            self.validation_data_task=utils.seperate_DataFrame(self.validation_data,self.n_validation_sample)

    
    def __repr__(self):
        text='#'*50+'\n'+'SISSO Classification\n'+'#'*50
        text+='\nSample Number: %d\nTask Number: %d\nRung: %d\nDimension: %d\nSubs_sis: %d\n'%(self.n_sample,self.n_task,self.rung,self.dimension,self.subs_sis)
        return text
    '''
    @property
    def baseline(self):
        r"""
        Returns the baseline, i.e., the errors with predicting every property using the mean value of the property in ``train.dat``.
        """
        
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
    '''
    def get_descriptors(self,path=None):
        r"""
        Returns descriptors.
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
    
    @property
    def descriptors(self):
        r"""
        Returns descriptors.
        """
        
        return self.get_descriptors()
    
    def features_percent(self,descending=True):
        r"""
        Computes the percentages of each feature in top subs_sis 1D descriptors.
        """
        
        feature_space=pd.read_csv(os.path.join(self.current_path,'feature_space','Uspace.name'),sep=r'\s+',header=None).iloc[0:self.subs_sis,0]
        feature_percent=pd.DataFrame(columns=self.features_name,index=('percent',))
        for feature_name in self.features_name:
            percent=feature_space.str.contains(feature_name).sum()/self.subs_sis
            feature_percent.loc['percent',feature_name]=percent
        feature_percent=feature_percent.T
        if descending:
            feature_percent.sort_values('percent',ascending=False,inplace=True)
        return feature_percent
    
    def evaluate_expression(self,expression,data=None):
        r"""
        Returns the value computed using given expression over data in ``train.dat``.
        
        Returns the values using ``expression`` to compute ``data``.
    
        Arguments:
            expression (string):  
                arithmetic expression. 
                The expression should be in the same form as the descriptor in file ``SISSO.out``,
                i.e., every operation should be enclosed in parenthesis, like (exp((A+B))/((C*D))^2).
            
            data (pandas.DataFrame):  
                the data you want to compute, with samples as index and features as columns.
                The features name should be correspondent to the operands in expression.
        
        Returns:
            pandas.Series: values computed using ``expression``
        """
        
        if isinstance(data,str):
            data=pd.read_csv(os.path.join(data),sep=r'\s+')
        if data == None:
            data=self.data
        return evaluate_expression(expression,data)
    '''
    def predict(self,data,tasks=None,dimensions=None):
        r"""
        Returns the predictions of ``data`` using the models found by SISSO.
        
        Arguments:
            data (string or pandas.DataFrame):  
                the path of the data or the data you want to compute. 
                If it is :class:`string`, it will be recognized as path to the data and 
                use ``data=pd.read_csv(data,sep=r'\s+')`` to read the data, so remember to use space to seperate the data.
                Otherwise it should be :class:`pandas.DataFrame`.
            
            tasks (None or list):  
                specifies which sample should use which task of model to compute.
                It should be task=[ [task_index, [sample_indices] ] ].
                For example, [ [1, [1,3]], [2, [2,4,5]] ] means sample 1 and 3 will be computed using task 1 model,
                and sample 2, 4 and 5 will be computed using task 2 model.
                If it is None, then compute all samples with task 1 model.
            
            dimensions (None or list):  
                specifies which dimension of desctiptor will be used.
                For example, [2,5] means only compute using 2D and 5D models.
                If it is None, then compute all samples with all dimension models.
        
        Returns:
            Values computed using models found by SISSO.
        """
        
        if isinstance(data,str):
            data=pd.read_csv(os.path.join(data),sep=r'\s+')
        return predict(data,self.descriptors,self.coefficients,self.intercepts,tasks=tasks,dimensions=dimensions)
    
    def predictions(self,training=True,display_task=False):
        r"""
        Returns predictions.
        The index order is dimension, sample if ``display_task`` is ``False``,
        or task, sample, dimension if ``display_task`` is ``True``.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool): 
                determines the predictions contain index of task or not.
        
        Returns:
            Predictions.
        """
        
        if display_task==True:
            return compute_using_model(result=self,training=training)
        else:
            return np.hstack(compute_using_model(result=self,training=training))
    
    def errors(self,training=True,display_task=False):
        r"""
        Returns errors.
        The index order is dimension, sample if ``display_task`` is ``False``,
        or task, sample, dimension if ``display_task`` is ``True``.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the errors contain index of task or not.
        
        Returns:
            Errors.
        """
        
        if display_task==True:
            pred=self.predictions(training=training,display_task=True)
            return [pred[task]-self.property_task[task].values for task in range(0,self.n_task)]
        else:
            if training==True:
                return self.predictions(training=training,display_task=False)-self.property.values
            else:
                return self.predictions(training=training,display_task=False)-self.validation_data.iloc[:,1].values
    
    def total_errors(self,training=True,display_task=False,display_baseline=False):
        r"""    
        Compute errors.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the errors contain index of task or not.
            
            display_baseline (bool):  
                determines whether display baseline
            
        Returns:
            RMSE, MAE, 25%ile AE, 50%ile AE, 75%ile AE, 95%ile AE, MaxAE of given errors.
        """
        
        if display_baseline:
            return pd.concat([pd.DataFrame(self.baseline[1:].values,
                                        columns=['Baseline'],
                                        index=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE']).T,
                            compute_errors(self.errors(training=training,display_task=display_task))])
        else:
            return compute_errors(self.errors(training=training,display_task=display_task))
        
    def check_predictions(self,dimension,multiply_coefficients=True):
        r"""
        Checks predictions of each descriptor.
        
        Arguments:  
            dimension (int):
                dimension of the descriptor.
                
            multiply_coefficients (bool):  
                whether it should be multiplied by coefficients.
            
        Returns:
            Predictions of each descriptor.
        """
        
        descriptors=self.descriptors
        coefficients=self.coefficients
        n_sample=self.n_sample
        data=self.data_task
        predictions=np.zeros([dimension+1,np.array(n_sample).sum()])
        total_n_sample=0
        for task in range(len(n_sample)):
            predictions[0,total_n_sample:total_n_sample+n_sample[task]]=self.intercepts[task][dimension-1]
            total_n_sample+=n_sample[task]
        if multiply_coefficients:
            for d in range(dimension):
                total_n_sample=0
                for task in range(len(n_sample)):
                    predictions[d+1,total_n_sample:total_n_sample+n_sample[task]]=coefficients[task][dimension-1][d]*evaluate_expression(descriptors[dimension-1][d],data[task]).values
                    total_n_sample+=n_sample[task]
        else:
            for d in range(dimension):
                total_n_sample=0
                for task in range(len(n_sample)):
                    predictions[d+1,total_n_sample:total_n_sample+n_sample[task]]=evaluate_expression(descriptors[dimension-1][d],data[task]).values
                    total_n_sample+=n_sample[task]
        return predictions
    
    def check_percentage(self,dimension,absolute=True):
        r"""
        Checks percentage of each descriptors.
        
        Arguments:  
            dimension (int):
                dimension of the descriptor.
                
            absolute (bool):
                whether return the absolute descriptor.
                If it is ``True``, the numerator is absolute value of each descriptor multiplied by corresponding coefficient,
                and denominator is the sum of intercept and every descriptor multiplied by coefficient.
                If it is ``False``, the numerator is descriptor multiplied by corresponding coefficient,
                and denominator is the proeprty. In this case, sometimes it will be larger than 1.
                
        Returns:
            percentage which index is [dimension, sample].
        """
        
        predictions=self.check_predictions(dimension)
        if absolute:
            return np.abs(predictions)/np.sum(np.abs(predictions),axis=0)
        else:
            return predictions/self.property.values
    '''






class ClassificationCV(Classification):
    r"""
    Basic class for evaluating the cross validation results of classification.
    You should instantiate it first if you want to analyze CV results.
    You can use index to select a specific result from total CV results, and it will return
    a instance of :class:`Classification`.
    
    Arguments:
        current_path (string):  
            path to the directory of the cross validation results.
        
        property_name (string):  
            specifies the property name of your CV results.
        
        drop_index (list):  
            specifies which CV results you don't want to consider.
    
    .. note::
        You should use code in ``cross_validation.py`` to generate the CV files, otherwise the format
        may be wrong. 
        
        Its attributes contain all the input arguments in ``SISSO.in``, information about ``train.dat``,
        ``validation.dat``, and ``SISSO.out``.
    """
    
    def __init__(self,current_path,property_name=None,drop_index=[]):
        # -------------------------- read path and CV info----------------------------
        cv_names=sorted(list(filter(lambda x: '_cv' in x,os.listdir(current_path))),
                        key=lambda x:int(x.split('_cv')[-1]))
        dir_list=list(map(lambda cv_name:os.path.join(current_path,cv_name),cv_names))
        n_cv=len(cv_names)
        
        self.current_path=current_path
        self.cv_path=[dir_list[cv] for cv in range(n_cv) if cv not in drop_index]
        self.drop_index=drop_index
        self.n_cv=n_cv-len(drop_index)
        
        with open(os.path.join(self.current_path,'cross_validation_info.dat'),'r') as f:
            cv_info=json.load(f)
        self.cross_validation_type=cv_info['cross_validation_type']
        
        # ------------------- read arguments of SISSO.in ----------------------------
        with open(os.path.join(self.cv_path[0],'SISSO.in'),'r') as f:
            input_file=f.read()
            
            # keywords for the target properties
            n_task=int(re.findall(r'ntask\s*=\s*(\d+)',input_file)[0])
            dimension=int(re.findall(r'desc_dim\s*=\s*(\d+)',input_file)[0])
            
            # keywords for feature construction and sure independence screening
            rung=int(re.findall(r'rung\s*=\s*(\d+)',input_file)[0])
            operation_set=re.findall(r"opset\s*=\s*'(.+)'",input_file)
            operation_set=re.split(r'[\(\)]+',operation_set[0])[1:-1]
            maxcomplexity=int(re.findall(r'maxcomplexity\s*=\s*(\d+)',input_file)[0])
            dimclass=re.findall(r'dimclass=([\(\d:\)]+)',input_file)[0]
            maxfval_lb=float(re.findall(r'maxfval_lb\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            maxfval_ub=float(re.findall(r'maxfval_ub\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            subs_sis=int(re.findall(r'subs_sis\s*=\s*(\d+)',input_file)[0])
            
            # keywords for descriptor
            method=re.findall(r'method\s*=\s*\'(\w+)',input_file)[0]
            isconvex=re.findall(r'isconvex\s*=\s*([\(\)\d,\s]*)\)',input_file)[0]
            isconvex=re.split(r'\(|\s',isconvex)
            isconvex=list(filter(lambda x: bool(re.search(r'\d',x)),isconvex))
            isconvex=[list(map(lambda d: int(d),filter(lambda x: bool(re.search(r'\d',x)),re.split(r',|\s|\)',n)))) for n in isconvex]   
            width=float(re.findall(r'width\s*=\s*(\d+.\d+|\d+[eE]-?\d+)',input_file)[0])
            nm_output=int(re.findall(r'nm_output\s*=\s*(\d+)',input_file)[0])
        
        # keywords for the target properties
        self.n_task=n_task
        self.dimension=dimension
        
        # keywords for feature construction and sure independence screening
        self.rung=rung
        self.operation_set=operation_set
        self.maxcomplexity=maxcomplexity
        self.dimclass=dimclass
        self.maxfval_lb=maxfval_lb
        self.maxfval_ub=maxfval_ub
        self.subs_sis=subs_sis
        
        # keywords for descriptor
        self.method=method
        self.isconvex=isconvex
        self.width=width
        self.nm_output=nm_output
        
        # --------------------------- read total data -----------------------------
        
        self.total_data=pd.read_csv(os.path.join(current_path,'train.dat'),sep=r'\s+')
        self.n_total_sample=len(pd.read_csv(os.path.join(current_path,'train.dat'),sep=r'\s+'))
        self.property_name=property_name if property_name else cv_names[0].split('_cv')[0]
        
        # -------------------------- read data per CV -----------------------------
        
        self.data=[pd.read_csv(os.path.join(cv_path,'train.dat'),sep=r'\s+') for cv_path in self.cv_path]
        self.materials=[data.iloc[:,0] for data in self.data]
        self.features_name=self.data[0].columns.tolist()[2:]
        self.n_sample=np.array([json.load(open(os.path.join(cv_path,'shuffle.dat'),'r'))['training_samples_number']
                        for cv_path in self.cv_path])
        self.n_validation_sample=np.array([json.load(open(os.path.join(cv_path,'shuffle.dat'),'r'))['validation_samples_number']
                        for cv_path in self.cv_path])
        self.validation_data=[pd.read_csv(os.path.join(cv_path,'validation.dat'),sep=r'\s+') for cv_path in self.cv_path]
        
        # --------------------- read data per CV per task ------------------------
        
        self.data_task=[utils.seperate_DataFrame(self.data[cv],self.n_sample[cv])
                        for cv in range(0,self.n_cv)]
        self.materials_task=[utils.seperate_DataFrame(self.materials[cv],self.n_sample[cv])
                        for cv in range(0,self.n_cv)]
        self.validation_data_task=[utils.seperate_DataFrame(self.validation_data[cv],self.n_validation_sample[cv])
                        for cv in range(0,self.n_cv)]
    
    def __getitem__(self,index):
        if isinstance(index,slice):
            return [Classification(self.cv_path[i]) for i in range(index.start,index.stop)]
        else:
            return Classification(self.cv_path[index])
    
    def __repr__(self):
        text='#'*50+'\n'+'SISSO Classification CV\n'+'#'*50
        with open(os.path.join(self.current_path,'cross_validation_info.dat'),'r') as f:
            cv_info=json.load(f)
        if 'shuffle_data_list' in cv_info:
            text+=('\nCross Validation Type: %s\nShuffle Data List: '%cv_info['cross_validation_type']+str(cv_info['shuffle_data_list']))
        else:
            text+=('\nCross Validation Type: %s\nIteration: '%cv_info['cross_validation_type']+str(self.n_cv))
        text+='\nTask Number: %d\nRung: %d\nDimension: %d\nSubs_sis: %d'%(self.n_task,self.rung,self.dimension,self.subs_sis)
        return text
    '''
    @property
    def baseline(self):
        r"""
        Returns the baseline, i.e., the errors with predicting every property using the mean value of the property in ``train.dat``.
        """
        
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
    '''
    def find_materials_in_validation(self,*idxs):
        r"""
        Returns the samples' names and in which CV result they are.
        
        Arguments:
            idx (integer):  
                index of the sample.
        
        Returns:
            the samples' names and in which CV result they are.
        """
        
        if self.cross_validation_type.startswith('leave'):
            val_num=len(self.validation_data[0])
            return ([self.validation_data[int(index/val_num)].iloc[index%val_num,0] for index in idxs],
                    [int(self.cv_path[int(index/val_num)].split('cv')[-1]) for index in idxs])
    '''
    def find_max_error(self):
        r"""
        Returns the samples' names and in which CV result they are that contribute to the maxAE.
        """
        
        return self.find_materials_in_validation(*(np.argmax(np.abs(self.errors(training=False)),axis=1)))
    '''
    @property
    def descriptors(self):
        r"""
        Returns descriptors. The first index is cross validation index.
        """
        
        return [super(ClassificationCV,self).get_descriptors(path=cv_path) for cv_path in self.cv_path]
    
    def features_percent(self,descending=True):
        r"""
        Return the percent of each feature in the top subs_sis descriptors.
        There are total n_cv*subs_sis descriptors,
        the feature percent is the percent over these descriptors.
        """
        
        feature_percent=pd.DataFrame(columns=self.features_name,index=('percent',))
        feature_percent.iloc[0,:]=0
        for cv_path in self.cv_path:
            feature_space=pd.read_csv(os.path.join(cv_path,'feature_space','Uspace.name'),sep=r'\s+',header=None).iloc[0:self.subs_sis,0]
            for feature_name in self.features_name:
                count=feature_space.str.contains(feature_name).sum()
                feature_percent.loc['percent',feature_name]+=count
        feature_percent.iloc[0,:]=feature_percent.iloc[0,:]/(self.n_cv*self.subs_sis)
        feature_percent=feature_percent.T
        if descending:
            feature_percent.sort_values('percent',ascending=False,inplace=True)
        return feature_percent
    
    def descriptor_percent(self,descriptor):
        r"""
        Return the percent of given descriptor appearing in the cross validation top subs_sis descriptors,
        and return the appearing index in the descriptor space.
        
        Arguments:
            descriptor (string):  
                the descriptor you want to check.
        """
        
        count=0
        descriptor_index=np.zeros(self.n_cv)
        for cv in range(0,self.n_cv):
            feature_space=pd.read_csv(os.path.join(self.cv_path[cv],'feature_space','Uspace.name'),sep=r'\s+',header=None).iloc[0:self.subs_sis,0]
            try:
                descriptor_index[cv]=feature_space.tolist().index(descriptor)+1
                count+=1
            except ValueError:
                descriptor_index[cv]=None
        return count/self.n_cv,descriptor_index
    '''
    def predict(self,data,cv_index=None,tasks=None,dimensions=None):
        r"""
        Returns the predictions of ``data`` using the models found by SISSO.
        
        Arguments:
            data (string or pandas.DataFrame):  
                the path of the data or the data you want to compute. 
                If it is :class:`string`, it will be recognized as path to the data and 
                use ``data=pd.read_csv(data,sep=r'\s+')`` to read the data, so remember to use space to seperate the data.
                Otherwise it should be :class:`pandas.DataFrame`.
            
            cv_index (None or list): 
                specifies which CV should be included.
                For example, [1,5] means CV1 and CV5 will be included.
                If it is None, then compute all CV results.
            
            tasks (None or list):  
                specifies which sample should use which task of model to compute.
                It should be task=[ [task_index, [sample_indices] ] ].
                For example, [ [1, [1,3]], [2, [2,4,5]] ] means sample 1 and 3 will be computed using task 1 model,
                and sample 2, 4 and 5 will be computed using task 2 model.
                If it is None, then compute all samples with task 1 model.
            
            dimensions (None or list):  
                specifies which dimension of desctiptor will be used.
                For example, [2,5] means only compute using 2D and 5D models.
                If it is None, then compute all samples with all dimension models.
        
        Returns:
            Values computed using models found by SISSO.
        """
        
        if cv_index==None:
            cv_index=list(range(self.n_cv))
        
        return [predict_reg(data,self.descriptors[cv],self.coefficients[cv],self.intercepts[cv],tasks=tasks,dimensions=dimensions)
                for cv in cv_index]
    
    def predictions(self,training=True,display_cv=False,display_task=False):
        r"""
        Returns predictions.
        The index order is dimension, sample if ``display_cv`` is ``False`` and ``display_task`` is ``False``,
        or task, sample, dimension if ``display_cv`` is ``False`` and ``display_task`` is ``True``,
        or CV index, sample, dimension if ``display_cv`` is ``True`` and ``display_task`` is ``False``,
        or CV index, task, sample, dimension if ``display_cv`` is ``True`` and ``display_task`` is ``True``.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the predictions contain index of task or not.
        
        Returns:
            Predictions.
        """
        
        if display_cv==True:
            if display_task==True:
                return [compute_using_model_reg(path=cv_path,
                                        training=training)
                for cv_path in self.cv_path]
            else:
                return [np.hstack(compute_using_model_reg(path=cv_path,
                                        training=training))
                for cv_path in self.cv_path]
        else:
            return np.hstack(self.predictions(training=training,display_cv=True,display_task=False))
    
    def errors(self,training=True,display_cv=False,display_task=False):
        r"""
        Returns errors.
        The index order is dimension, sample if ``display_cv`` is ``False`` and ``display_task`` is ``False``,
        or task, sample, dimension if ``display_cv`` is ``False`` and ``display_task`` is ``True``,
        or CV index, sample, dimension if ``display_cv`` is ``True`` and ``display_task`` is ``False``,
        or CV index, task, sample, dimension if ``display_cv`` is ``True`` and ``display_task`` is ``True``.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_task (bool):  
                determines the errors contain index of task or not.
        
        Returns:
            Errors.
        """
        
        if display_cv:
            if training:
                if display_task:
                    error=[]
                    pred=self.predictions(training=True,display_cv=True,display_task=True)
                    for cv in range(self.n_cv):
                        error_cv=[]
                        for task in range(0,self.n_task):
                            error_cv.append(pred[cv][task]-self.property_task[cv][task])
                        error.append(error_cv)
                    return error
                else:
                    pred=self.predictions(training=True,display_cv=True,display_task=True)
                    return [np.hstack(pred[cv])-np.hstack(self.property_task[cv])
                            for cv in range(self.n_cv)]
            else:
                if display_task:
                    error=[]
                    pred=self.predictions(training=False,display_cv=True,display_task=True)
                    for cv in range(0,self.n_cv):
                        error_cv=[]
                        for task in range(self.n_task):
                            error_cv.append(pred[cv][task]-self.validation_data_task[cv][task].iloc[:,1].values)
                        error.append(error_cv)
                    return error
                else:
                    pred=self.predictions(training=False,display_cv=True,display_task=True)
                    return [(np.hstack(pred[cv])-self.validation_data[cv].iloc[:,1].values)
                            for cv in range(self.n_cv)]
        else:
            if display_task:
                errors_cv_t=self.errors(training=training,display_cv=True,display_task=True)
                errors=[]
                for task in range(self.n_task):
                    errors_t=errors_cv_t[0][task]
                    for cv in range(1,self.n_cv):
                        errors_t=np.hstack((errors_t,errors_cv_t[cv][task]))
                    errors.append(errors_t)
                return errors
            else:
                return np.hstack(self.errors(training=training,display_cv=True,display_task=False))
        
    def total_errors(self,training=True,display_cv=False,display_task=False,display_baseline=False):
        r"""    
        Compute errors.
        
        Arguments:
            training (bool):  
                determines whether its training or not.
            
            display_cv (bool):  
                determines the errors contain index of CV or not.
            
            display_task (bool):  
                determines the errors contain index of task or not.
            
            display_baseline (bool):  
                determines whether display baseline
            
        Returns:
            RMSE, MAE, 25%ile AE, 50%ile AE, 75%ile AE, 95%ile AE, MaxAE of given errors.
        """
        
        if display_baseline:
            if training:
                return pd.concat([pd.DataFrame(self.baseline[1:].values,
                                        columns=['Baseline'],
                                        index=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE']).T,
                                compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task))])
            else:
                return pd.concat([pd.DataFrame(self.baseline[1:].values,
                                        columns=['Baseline'],
                                        index=['RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE']).T,
                                compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task))])
        else:
            if training:
                return compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task))
            else:
                return compute_errors(self.errors(training=training,display_cv=display_cv,display_task=display_task))
    '''
    def drop(self,index=[]):
        r"""
        Drop some CV results.
        
        Arguments:
            index (list):  
                contain the CV index which you want to drop.
        
        Returns:
            An instance of RegressionCV that drop the results in CV index.
        """
        
        index+=self.drop_index
        return ClassificationCV(self.current_path,self.property_name,index)
    '''
    def check_predictions(self,cv_idx,dimension,training=True,multiply_coefficients=True):
        r"""
        Check predictions of each descriptor.
        
        Arguments:
            cv_idx (integer):  
                specifies which CV result you want to check.
                
            dimension (int):
                dimension of the descriptor.
            
            training (bool):  
                whether it is training.
            
            multiply_coefficients (bool):  
                whether it should be multiplied by coefficients.
            
        Returns
            Predictions of each descriptor.
        """
        
        descriptors=self.descriptors[cv_idx]
        coefficients=self.coefficients[cv_idx]
        if training:
            n_sample=self.n_sample[cv_idx]
            data=self.data_task[cv_idx]
        else:
            n_sample=self.n_validation_sample[cv_idx]
            data=self.validation_data_task[cv_idx]
        predictions=np.zeros([n_sample.sum(),dimension+1])
        total_n_sample=0
        for task in range(len(n_sample)):
            predictions[total_n_sample:total_n_sample+n_sample[task],0]=self.intercepts[cv_idx][task][dimension-1]
            total_n_sample+=n_sample[task]
        if multiply_coefficients:
            for d in range(dimension):
                total_n_sample=0
                for task in range(len(n_sample)):
                    predictions[total_n_sample:total_n_sample+n_sample[task],d+1]=coefficients[task][dimension-1][d]*evaluate_expression(descriptors[dimension-1][d],data[task]).values
                    total_n_sample+=n_sample[task]
        else:
            for d in range(dimension):
                total_n_sample=0
                for task in range(len(n_sample)):
                    predictions[total_n_sample:total_n_sample+n_sample[task],d+1]=evaluate_expression(descriptors[dimension-1][d],data[task]).values
                    total_n_sample+=n_sample[task]
        return predictions
        
    def check_percentage(self,cv_idx,dimension,absolute=True):
        r"""
        Checks percentage of each descriptors.
        
        Arguments:  
            dimension (int):
                dimension of the descriptor.
                
            absolute (bool):
                whether return the absolute descriptor.
                If it is ``True``, the numerator is absolute value of each descriptor multiplied by corresponding coefficient,
                and denominator is the sum of intercept and every descriptor multiplied by coefficient.
                If it is ``False``, the numerator is descriptor multiplied by corresponding coefficient,
                and denominator is the proeprty. In this case, sometimes it will be larger than 1.
                
        Returns:
            percentage which index is [dimension, sample].
        """
        
        predictions=self.check_predictions(cv_idx,dimension)
        if absolute:
            return np.abs(predictions)/np.sum(np.abs(predictions),axis=0)
        else:
            return predictions/self.property[cv_idx].values
    '''
