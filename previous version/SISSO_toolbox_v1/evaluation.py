import pandas as pd
import numpy as np
import string
import re
import os
import sys
import functools
import math

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
                        OPND.append(np.power(operand,-1))
                    elif power=='^2':
                        OPND.append(np.power(operand,2))
                    elif power=='^3':
                        OPND.append(np.power(operand,3))
                    elif power=='^6':
                        OPND.append(np.power(operand,6))
                    OPTR.pop()
                elif len(OPTR)>1 and OPTR[-1]=='(':
                    operand=OPND.pop()
                    if OPTR[-2]=='exp':
                        OPND.append(np.exp(operand))
                    elif OPTR[-2]=='exp-':
                        OPND.append(np.exp(-operand))
                    elif OPTR[-2]=='sqrt':
                        OPND.append(np.sqrt(operand))
                    elif OPTR[-2]=='cbrt':
                        OPND.append(np.cbrt(operand))
                    elif OPTR[-2]=='log':
                        OPND.append(np.log(operand))
                    elif OPTR[-2]=='abs':
                        OPND.append(np.abs(operand))
                    elif OPTR[-2]=='sin':
                        OPND.append(np.sin(operand))
                    elif OPTR[-2]=='cos':
                        OPND.append(np.cos(operand))
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


def compute_using_descriptors(path=None,data_path=None,result=None):
    """
    Return a 2D numpy array (sample_index, dimension),
    whose item is the value computed using descriptors found by SISSO.
    """
    if path:
        result=Result(path)
    if data_path:
        data=pd.read_csv(data_path,sep=' ')
    else:
        data=result.data
    pred=[]
    for dimension in range(0,result.dimension):
        value=0
        for i in range(0,dimension+1):
            value+=result.coefficients()[dimension][i]*evaluate_expression(result.descriptors()[dimension][i],data)
        value+=result.intercepts()[dimension]
        pred.append(np.array(value))
    return np.array(pred).T

def errors(path=None,data_path=None,result=None):
    """
    Return the errors of given class Result's data or a given data path.
    """
    if path:
        result=Result(path)
    if data_path:
        data=pd.read_csv(data_path,sep=' ')
    else:
        data=result.data
    samples_num=len(data)
    property_test=data.iloc[:,1]
    error=np.zeros(shape=(result.dimension,6))
    for dimension in range(0,result.dimension):
        value=0
        for i in range(0,dimension+1):
            value+=result.coefficients()[dimension][i]*evaluate_expression(result.descriptors()[dimension][i],data)
        value+=result.intercepts()[dimension]
        sorted_error=np.sort(np.abs(value-property_test))
        error[dimension]=np.array([np.sqrt(np.mean(np.power(sorted_error,2))),
                            np.mean(sorted_error),
                            sorted_error[math.ceil(samples_num*0.5)-1],
                            sorted_error[math.ceil(samples_num*0.75)-1],
                            sorted_error[math.ceil(samples_num*0.95)-1],
                            sorted_error[-1]])
    error=pd.DataFrame(error,columns=['RMSE','MAE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],index=list(range(1,result.dimension+1)))
    return error

def item_errors(item_index,path=None,data_path=None,result=None):
    """
    Return the errors of item_index data point in the data set.
    """
    if path:
        result=Result(path)
    if data_path:
        data=pd.read_csv(data_path,sep=' ')
    else:
        data=result.data
    data=data.iloc[item_index]
    property_test=data.iloc[1]
    error=np.zeros(result.dimension)
    for dimension in range(0,result.dimension):
        value=0
        for i in range(0,dimension+1):
            value+=result.coefficients()[dimension][i]*evaluate_expression(result.descriptors()[dimension][i],data)
        value+=result.intercepts()[dimension]
        error[dimension]=value-property_test
    return error

def total_items_errors(errors):
    """
    Return the errors of given 2D numpy array errors (sample_index, dimension), if errors is 2D numpy array,
    or return the errors of given 1D numpy array error
    """
    samples_num=len(errors)
    if errors.ndim==1:
        error=np.zeros(shape=(1,6))
        sorted_error=np.sort(np.abs(errors))
        error[0]=np.array([np.sqrt(np.mean(np.power(sorted_error,2))),
                                np.mean(sorted_error),
                                sorted_error[math.ceil(samples_num*0.5)-1],
                                sorted_error[math.ceil(samples_num*0.75)-1],
                                sorted_error[math.ceil(samples_num*0.95)-1],
                                sorted_error[-1]])
        error=pd.DataFrame(error,columns=['RMSE','MAE','50%ile AE','75%ile AE','95%ile AE','MaxAE'])
    elif errors.ndim==2:
        dimensions=len(errors[0])
        error=np.zeros(shape=(dimensions,6))
        for dimension in range(0,dimensions):
            sorted_error=np.sort(np.abs(errors[:,dimension]))
            error[dimension]=np.array([np.sqrt(np.mean(np.power(sorted_error,2))),
                                np.mean(sorted_error),
                                sorted_error[math.ceil(samples_num*0.5)-1],
                                sorted_error[math.ceil(samples_num*0.75)-1],
                                sorted_error[math.ceil(samples_num*0.95)-1],
                                sorted_error[-1]])
        error=pd.DataFrame(error,columns=['RMSE','MAE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],index=list(range(1,dimensions+1)))
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
        self.operation_set=operation_set
        self.data=pd.read_csv(os.path.join(current_path,'train.dat'),sep=' ')
        self.property_name=self.data.columns.tolist()[1]
        self.property=self.data.iloc[:,1]
        self.features_name=self.data.columns.tolist()[2:]
        self.materials=self.data.iloc[:,0]
        self.samples_number=len(self.materials)
        self.subs_sis=subs_sis
        self.rung=rung
        self.dimension=dimension
    
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
        Return a list, whose ith item is ith D coefficients.
        """
        if path==None:
            path=self.current_path
        coefficients_all=[]
        with open(os.path.join(path,'SISSO.out'),'r') as f:
            input_file=f.read()
            coefficients_total=re.findall(r'coefficients_001:(.*)',input_file)
            for dimension in range(0,self.dimension):
                coefficients_d=re.split(r'\s+',coefficients_total[dimension])[1:]
                coefficients_d=list(map(float,coefficients_d))
                coefficients_all.append(coefficients_d)
        return coefficients_all
                
    
    def intercepts(self,path=None):
        """
        Return a list, whose ith item is ith D intercepts.
        """
        if path==None:
            path=self.current_path
        with open(os.path.join(path,'SISSO.out'),'r') as f:
            input_file=f.read()
            intercepts_all=re.findall(r'Intercept_001:(.*)',input_file)
            intercepts_all=list(map(float,list(map(str,intercepts_all))))
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
    
    def training_values(self):
        """
        Return a 2D numpy array (sample_index, dimension),
        whose item is the value computed using descriptors found by SISSO.
        """
        return compute_using_descriptors(result=self)
    
    """
    def training_errors_old(self):
        errors=np.zeros(shape=(self.dimension,2))
        with open(os.path.join(self.current_path,'SISSO.out'),'r') as f:
            input_file=f.read()
            errors=np.array(re.findall(r'\dD des.+\s+.+:\s+(\d*\.\d*)\s*(\d*\.\d*)',input_file))
        return errors
    """
    
    def training_errors(self):
        """
        Return a pandas DataFrame (dimension, type of error).
        """
        return errors(result=self)
    
    def items_training_errors(self):
        """
        Return a 2D numpy array (sample_index, dimension), whose value is error.
        """
        return np.array([item_errors(item_index,result=self) for item_index in range(0,self.samples_number)])





class Results(Result):
    """
    Evaluate the cross validation results of SISSO.
    """
    
    def __init__(self,current_path,property_name,cv_number):
        self.current_path=current_path
        self.property_name=property_name
        self.cv_number=cv_number
        
        with open(os.path.join(self.current_path,'%s_cv0'%(self.property_name),'SISSO.in'),'r') as f:
                input_file=f.read()
                subs_sis=int(re.findall(r'subs_sis\s*=\s*(\d+)',input_file)[0])
                rung=int(re.findall(r'rung\s*=\s*(\d+)',input_file)[0])
                dimension=int(re.findall(r'desc_dim\s*=\s*(\d+)',input_file)[0])
                operation_set=re.findall(r"opset\s*=\s*'(.+)'",input_file)
                operation_set=re.split(r'[\(\)]+',operation_set[0])[1:-1]
        self.operation_set=operation_set
        self.subs_sis=subs_sis
        self.rung=rung
        self.dimension=dimension
        self.total_materials_number=len(pd.read_csv(os.path.join(current_path,'train.dat'),sep=' '))
        self.data=[]
        self.materials=[]
        self.samples_number=[]
        self.property=[]
        self.validation_data=[]
        for cv in range(0,cv_number):
            self.data.append(pd.read_csv(os.path.join(current_path,'%s_cv%d'%(self.property_name,cv),'train.dat'),sep=' '))
            self.validation_data.append(pd.read_csv(os.path.join(current_path,'%s_cv%d'%(self.property_name,cv),'validation.dat'),sep=' '))
            self.property.append(pd.read_csv(os.path.join(current_path,'%s_cv%d'%(self.property_name,cv),'train.dat'),sep=' ').iloc[:,1])
            self.materials.append(self.data[cv].iloc[:,0])
            self.samples_number.append(len(self.materials[cv]))
        self.features_name=self.data[0].columns.tolist()[2:]
    
    def find_materials_in_validation(self,*idxs):
        val_num=len(self.validation_data[0])
        return [self.validation_data[int(index/val_num)].iloc[index%val_num,0] for index in idxs]
    
    def descriptors(self):
        """
        Return a list whose index refers to the index of corss validation.
        Each item in the list is also a list, whose jth item is j+1 D descriptors.
        """
        return [super(Results,self).descriptors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv))) for cv in range(0,self.cv_number)]
    
    def coefficients(self):
        """
        Return a list whose index refers to the index of corss validation.
        Each item in the list is also a list, whose jth item is j+1 D coefficients.
        """
        return [super(Results,self).coefficients(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv))) for cv in range(0,self.cv_number)]
    
    def intercepts(self):
        """
        Return a list whose index refers to the index of corss validation.
        Each item in the list is also a list, whose jth item is j+1 D intercepts.
        """
        return [super(Results,self).intercepts(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv))) for cv in range(0,self.cv_number)]
    
    def features_percent(self):
        """
        Return the percent of each feature in the top subs_sis descriptors.
        There are total cv_number*subs_sis descriptors,
        the feature percent is the percent over these descriptors.
        """
        """
        feature_percent=pd.DataFrame(self.features_name)
        feature_percent.insert(1,'percent',np.zeros(len(self.features_name)))
        for cv in range(0,self.cv_number):
            feature_space=pd.read_csv(os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'feature_space','Uspace.name'),sep=' ',header=None).iloc[0:self.subs_sis,0]
            index=0
            for feature_name in self.features_name:
                count=feature_space.str.contains(feature_name).sum()
                feature_percent.loc[index,'percent']+=count
                index+=1
        feature_percent.iloc[:,1]=feature_percent.iloc[:,1]/(self.cv_number*self.subs_sis)
        feature_percent.sort_values('percent',inplace=True,ascending=False)
        return feature_percent
        """
        feature_percent=pd.DataFrame(columns=self.features_name,index=('percent',))
        feature_percent.iloc[0,:]=0
        for cv in range(0,self.cv_number):
            feature_space=pd.read_csv(os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'feature_space','Uspace.name'),sep=' ',header=None).iloc[0:self.subs_sis,0]
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
            feature_space=pd.read_csv(os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'feature_space','Uspace.name'),sep=' ',header=None).iloc[0:self.subs_sis,0]
            try:
                descriptor_index[cv]=feature_space.tolist().index(descriptor)+1
                count+=1
            except ValueError:
                descriptor_index[cv]=None
        return count/self.cv_number,descriptor_index
    
    def training_values(self):
        """
        Return a list, whose ith item is ith cross validation value computed by corresponding descriptors found by SISSO.
        Each item is a 2D numpy array (sample_index, dimension).
        """
        return [compute_using_descriptors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)))
                for cv in range(0,self.cv_number)]
        
    def prediction_values(self):
        """
        Return a list, whose ith item is ith cross validation value computed by corresponding descriptors found by SISSO.
        Each item is a 2D numpy array (sample_index, dimension).
        """
        return [compute_using_descriptors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)),
                                        data_path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'validation.dat'))
                for cv in range(0,self.cv_number)]
    
    def training_errors(self):
        """
        Return a list, whose ith item is a pandas DataFrame,
        which refers to ith cross validation errors.
        """
        return [errors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)))
                for cv in range(0,self.cv_number)]
    
    def items_training_errors(self):
        """
        Return a 2D numpy array (sample_index, dimension),
        whose item is the error corresponding to the sample.
        """
        return np.array([item_errors(item_index=item_index,
                        path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)))
                        for cv in range(0,self.cv_number) for item_index in range(0,self.samples_number[cv])])
    
    def prediction_errors(self):
        """
        Return a list, whose ith item is a pandas DataFrame,
        which refers to ith cross validation errors.
        """
        return [errors(path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)),
                        data_path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'validation.dat'))
                for cv in range(0,self.cv_number)]
        
    def items_prediction_errors(self):
        """
        Return a 2D numpy array (sample_index, dimension),
        whose item is the error corresponding to the sample.
        """
        return np.array([item_errors(item_index=item_index,
                        path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv)),
                        data_path=os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'validation.dat'))
                        for cv in range(0,self.cv_number) for item_index in range(0,self.total_materials_number-self.samples_number[cv])])
    
    def total_training_errors(self):
        """
        Return the errors over whole cross validation.
        """
        return total_items_errors(self.items_training_errors())
    
    def total_prediction_errors(self):
        """
        Return the errors over whole cross validation.
        """
        return total_items_errors(self.items_prediction_errors())
    
    """
    def training_errors_old(self):
        errors=np.zeros(shape=(self.cv_number,self.dimension,2))
        for cv in range(0,self.cv_number):
            with open(os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'SISSO.out'),'r') as f:
                input_file=f.read()
                errors[cv]=np.array(re.findall(r'\dD des.+\s+.+:\s+(\d*\.\d*)\s*(\d*\.\d*)',input_file))
        return errors

    def prediction_errors_old(self):
        errors=[]
        for cv in range(0,self.cv_number):
            with open(os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'shuffle.dat'),'r') as f:
                input_file=f.read()
                val_list=re.findall(r'validation_list:\ \[[(\d+)\s+]+',input_file)[0]
                val_list=re.split(r'\s+|\[',val_list)
                val_list=np.array(list(map(int,list(filter(str.isdigit,val_list)))))
            data=pd.read_csv(os.path.join(self.current_path,'train.dat'),sep=' ').iloc[val_list-1]
            property_test=data.iloc[:,1]
            with open(os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'SISSO.out'),'r') as f:
                input_file=f.read()
                descriptors=re.findall(r'descriptor:[\s\S]*?coefficients',input_file)
                coefficients=re.findall(r'coefficients_001:(.*)',input_file)
                intercepts=re.findall(r'Intercept_001:(.*)',input_file)
                intercepts=list(map(float,list(map(str,intercepts))))
                ers=np.zeros(shape=(self.dimension,6))
                for dimension in range(0,self.dimension):
                    descriptors_d=descriptors[dimension]
                    descriptors_d=re.split(r'\s+',descriptors_d)
                    descriptors_d=descriptors_d[1:dimension+2]
                    descriptors_d=[x[1] for x in list(map(lambda x: re.split(r':',x),descriptors_d))]
                    #descriptors_d=list(map(lambda x: x.strip(r'[]'),descriptors_d))
                    descriptors_d=list(map(lambda x: x.replace(r'[',r'('),descriptors_d))
                    descriptors_d=list(map(lambda x: x.replace(r']',r')'),descriptors_d))
                    coefficients_d=re.split(r'\s+',coefficients[dimension])[1:]
                    coefficients_d=list(map(float,coefficients_d))
                    value=0
                    for i in range(0,dimension+1):
                        value+=coefficients_d[i]*evaluate_expression(descriptors_d[i],data)
                    value+=intercepts[dimension]
                    ers[dimension]=np.array([np.sqrt(np.mean((value-property_test)**2)),
                                        np.mean(np.abs(value-property_test)),
                                        np.sort(np.abs(value-property_test))[math.ceil(len(data)*0.5)-1],
                                        np.sort(np.abs(value-property_test))[math.ceil(len(data)*0.75)-1],
                                        np.sort(np.abs(value-property_test))[math.ceil(len(data)*0.95)-1],
                                        np.sort(np.abs(value-property_test))[-1]])
            ers=pd.DataFrame(ers,columns=['RMSE','MAE','50%ile AE','75%ile AE','95%ile AE','MaxAE'],index=list(range(1,self.dimension+1)))
            errors.append(ers)
        return errors
    
    def items_prediction_errors_old(self):
        error=[]
        for cv in range(0,self.cv_number):
            with open(os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'shuffle.dat'),'r') as f:
                input_file=f.read()
                val_list=re.findall(r'validation_list:\ \[[(\d+)\s+]+',input_file)[0]
                val_list=re.split(r'\s+|\[',val_list)
                val_list=np.array(list(map(int,list(filter(str.isdigit,val_list)))))
            data=pd.read_csv(os.path.join(self.current_path,'train.dat'),sep=' ').iloc[val_list-1]
            property_test=data.iloc[:,1]
            with open(os.path.join(self.current_path,'%s_cv%d'%(self.property_name,cv),'SISSO.out'),'r') as f:
                input_file=f.read()
                descriptors=re.findall(r'descriptor:[\s\S]*?coefficients',input_file)
                coefficients=re.findall(r'coefficients_001:(.*)',input_file)
                intercepts=re.findall(r'Intercept_001:(.*)',input_file)
                intercepts=list(map(float,list(map(str,intercepts))))
                ers=np.zeros(shape=(len(val_list),self.dimension))
                for dimension in range(0,self.dimension):
                    descriptors_d=descriptors[dimension]
                    descriptors_d=re.split(r'\s+',descriptors_d)
                    descriptors_d=descriptors_d[1:dimension+2]
                    descriptors_d=[x[1] for x in list(map(lambda x: re.split(r':',x),descriptors_d))]
                    #descriptors_d=list(map(lambda x: x.strip(r'[]'),descriptors_d))
                    descriptors_d=list(map(lambda x: x.replace(r'[',r'('),descriptors_d))
                    descriptors_d=list(map(lambda x: x.replace(r']',r')'),descriptors_d))
                    coefficients_d=re.split(r'\s+',coefficients[dimension])[1:]
                    coefficients_d=list(map(float,coefficients_d))
                    value=0
                    for i in range(0,dimension+1):
                        value+=coefficients_d[i]*evaluate_expression(descriptors_d[i],data)
                    value+=intercepts[dimension]
                    ers[:,dimension]=np.array(value-property_test)
            error.append(ers)
        return np.array(error)
    """