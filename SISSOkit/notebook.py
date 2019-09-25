from . import evaluation as evl
from . import utils as utils

import os
import re
import json
import string





def generate_report(path,file_path,notebook_name,file_name=None):
    r"""
    Generates jupyter notebook reports.
    
    Arguments:
        path (list):  
            path to SISSO results. If there is only one result over whole data set, then
            it should be a list containing only 1 item. If there is also cross validation results,
            it should be [path to result over whole data set, path to cross validation results].
        
        file_path (string):  
            path to newly generated jupyter notebook.
        
        notebook_name (int or string):  
            notebook index or notebook name.
            =====  =====
            index   name
            =====  =====
            0      regression
            1      regression with CV
            =====  =====
        
        file_name (None or string): the newly generated jupyter notebook name. If it is ``None``,
            the file name is the same as notebook template name.
    """
    
    if notebook_name=='regression' or notebook_name==0:
        notebook_name='regression'
        if file_name==None:
            file_name=notebook_name
        regression=evl.Regression(path[0])
        with open(os.path.join(os.path.dirname(__file__),'notebook_templates',notebook_name+'.ipynb'),'r') as f:
            notebook=json.load(f)
        notebook['cells'][2]['source'][0]='path="%s"\n'%path[0]
        
        for dimension in range(regression.dimension):
            notebook['cells'][11]['source'].append('1. %dD descriptor:\n'%(dimension+1))
            notebook['cells'][11]['source'].append('\n')
            for i in range(dimension+1):
                notebook['cells'][11]['source'].append('$$\n'+utils.descriptors_to_markdown(regression.descriptors[dimension][i])+'\n$$')
                notebook['cells'][11]['source'].append('\n')
                notebook['cells'][11]['source'].append('\n')
                notebook['cells'][11]['source'].append('$$$$\n')
                notebook['cells'][11]['source'].append('\n')
        
        for dimension in range(regression.dimension):
            notebook['cells'][28]['source'].append('1. %dD model:\n'%(dimension+1))
            notebook['cells'][28]['source'].append('\n')
            for task in range(regression.n_task):
                notebook['cells'][28]['source'].append('\t1. Task %d:\n'%(task+1))
                notebook['cells'][28]['source'].append('\n')
                notebook['cells'][28]['source'].append('$$\n'+utils.models_to_markdown(regression,task+1,dimension+1,indent='\\\\')+'\n$$')
                notebook['cells'][28]['source'].append('\n')
        
        with open(os.path.join(file_path,file_name+'.ipynb'),'w') as f:
            json.dump(notebook,f,indent=1)
    
    if notebook_name=='regression with CV' or notebook_name==1:
        notebook_name='regression with CV'
        if file_name==None:
            file_name=notebook_name
        regression=evl.Regression(path[0])
        with open(os.path.join(os.path.dirname(__file__),'notebook_templates',notebook_name+'.ipynb'),'r') as f:
            notebook=json.load(f)
        notebook['cells'][2]['source'][0]='path="%s"\n'%path[0]
        notebook['cells'][2]['source'][1]='cv_path="%s"\n'%path[1]
        
        for dimension in range(regression.dimension):
            notebook['cells'][13]['source'].append('1. %dD descriptor:\n'%(dimension+1))
            notebook['cells'][13]['source'].append('\n')
            for i in range(dimension+1):
                notebook['cells'][13]['source'].append('$$\n'+utils.descriptors_to_markdown(regression.descriptors[dimension][i])+'\n$$')
                notebook['cells'][13]['source'].append('\n')
                notebook['cells'][13]['source'].append('\n')
                notebook['cells'][13]['source'].append('$$$$\n')
                notebook['cells'][13]['source'].append('\n')
        
        for dimension in range(regression.dimension):
            notebook['cells'][41]['source'].append('1. %dD model:\n'%(dimension+1))
            notebook['cells'][41]['source'].append('\n')
            for task in range(regression.n_task):
                notebook['cells'][41]['source'].append('\t1. Task %d:\n'%(task+1))
                notebook['cells'][41]['source'].append('\n')
                notebook['cells'][41]['source'].append('$$\n'+utils.models_to_markdown(regression,task+1,dimension+1,indent='\\\\')+'\n$$')
                notebook['cells'][41]['source'].append('\n')
        
        with open(os.path.join(file_path,file_name+'.ipynb'),'w') as f:
            json.dump(notebook,f,indent=1)
