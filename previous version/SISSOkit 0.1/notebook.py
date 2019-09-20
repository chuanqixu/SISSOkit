import os
import re
import json
import string
from . import evaluation as evl
from . import utils as utils

def generate_report(path,file_path,notebook_name):
    
    regression=evl.Regression(path[0])
    with open(os.path.join(os.path.dirname(__file__),'notebook_template',notebook_name+'.ipynb'),'r') as f:
        notebook=json.load(f)
    notebook['cells'][2]['source'][0]='path="%s"\n'%path[0]
    notebook['cells'][2]['source'][1]='cv_path="%s"\n'%path[1]
    
    for dimension in range(regression.dimension):
        notebook['cells'][12]['source'].append('1. %dD model:\n'%(dimension+1))
        notebook['cells'][12]['source'].append('\n')
        for task in range(regression.n_task):
            notebook['cells'][12]['source'].append('\t1. Task %d:\n'%(task+1))
            notebook['cells'][12]['source'].append('\n')
            notebook['cells'][12]['source'].append('$$\n'+utils.models_to_markdown(regression,task+1,dimension+1,indent='\\\\')+'\n$$')
            notebook['cells'][12]['source'].append('\n')
    
    dimension=regression.dimension
    notebook['cells'][24]['source']=["plt.figure(figsize=(20,%d))\n"%(dimension*6),
    "for i in range(1,%d):\n"%(dimension+1),
    "    plt.subplot(%d,3,(i-1)*3+1)\n"%dimension,
    "    plot.error_hist(i,ST_cv,abs=False,training=False,rwidth=0.8)\n",
    "    plt.subplot(%d,3,(i-1)*3+2)\n"%dimension,
    "    plot.property_vs_prediction(i,ST_cv,training=False)\n",
    "    plt.subplot(%d,3,(i-1)*3+3)\n"%dimension,
    "    plot.hist_and_box_plot(i,ST_cv,training=False, bins=20, alpha=0.5, rwidth=0.8,marker_x=10)"]
    
    with open(os.path.join(file_path,notebook_name+'.ipynb'),'w') as f:
        json.dump(notebook,f,indent=1)