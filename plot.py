import matplotlib.pyplot as plt
import SISSO_analysis.evaluation as evl
import numpy as np
import functools

def baselineplot(result,unit_name=None,marker='x',marker_color='r',fontsize=20,marker_shape=100,marker_y=1,**kw):
    """
    Plot the baseline of given result.
    """
    plt.hist(result.property,zorder=0,**kw)
    plt.scatter(result.property.mean(),marker_y,marker=marker,c=marker_color,s=marker_shape,zorder=1,label='Mean Value')
    if unit_name:
        plt.xlabel(result.property_name+'['+unit_name+']',fontsize=fontsize)
    else:
        plt.xlabel(result.property_name,fontsize=fontsize)
    plt.ylabel('Counts',fontsize=fontsize)
    plt.title('Histogram of %s over training data\nMean = %.5f\nStandard deviation = %.5f'%(result.property_name,result.property.mean(),result.property.std()),fontsize=fontsize)
    plt.legend()

def signed_error_hist(dimension,*results,error_type='traning',unit_name=None,fontsize=20,**kw):
    """
    Plot the histogram of signed error
    """
    dimension-=1
    if error_type.startswith('t'):
        collect_data=np.hstack([result.items_training_errors()[:,dimension] for result in results])
    elif error_type.startswith('p'):
        collect_data=np.hstack([result.items_prediction_errors()[:,dimension] for result in results])
    plt.hist(collect_data,**kw)
    if unit_name:
        plt.xlabel('Signed error %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.xlabel('Signed error',fontsize=fontsize)
    plt.ylabel('Counts',fontsize=fontsize)
        
def absolute_error_hist(dimension,*results,error_type='traning',unit_name=None,fontsize=20,**kw):
    """
    Plot the histogram of absolute error
    """
    dimension-=1
    if error_type.startswith('t'):
        collect_data=np.hstack([np.abs(result.items_training_errors())[:,dimension] for result in results])
    elif error_type.startswith('p'):
        collect_data=np.hstack([np.abs(result.items_prediction_errors())[:,dimension] for result in results])
    plt.hist(collect_data,**kw)
    if unit_name:
        plt.xlabel('Absolute error %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.xlabel('Absolute error',fontsize=fontsize)
    plt.ylabel('Counts',fontsize=fontsize)
    
def property_vs_prediction(dimension,*results,error_type='traning',unit_name=None,fontsize=20,**kw):
    """
    Plot the scatter plot of property_vs_prediction
    """
    dimension-=1
    if isinstance(results[0],evl.Results):
        if error_type.startswith('t'):
            property_values=np.hstack([result.property[cv].values for result in results for cv in range(0,result.cv_number)])
            prediction_value=np.hstack([result.training_values()[cv][:,dimension] for result in results for cv in range(0,result.cv_number)])-property_values
        elif error_type.startswith('p'):
            property_values=np.hstack([result.validation_data[cv].iloc[:,1].values.tolist() for result in results for cv in range(0,result.cv_number)])
            prediction_value=np.hstack([result.prediction_values()[cv][:,dimension] for result in results for cv in range(0,result.cv_number)])-property_values
    else:
        property_values=np.hstack([result.property.values for result in results])
        prediction_value=np.hstack([result.training_values()[:,dimension] for result in results])-property_values
    plt.scatter(property_values,prediction_value,**kw)
    if unit_name:
        plt.xlabel('%s in data set %s'%(results[0].property_name,'['+unit_name+']'),fontsize=fontsize)
        plt.ylabel('Signed error %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.xlabel('%s in data set'%(results[0].property_name),fontsize=fontsize)
        plt.ylabel('Signed error',fontsize=fontsize)
        
def hist_and_box_plot(dimension,*results,error_type='traning',unit_name=None,fontsize=20,selected_errors=None,marker_x=0,marker=None,**kw):
    """
    Plot the histogram of absolute errors with box plot for errors
    """
    dimension-=1
    if error_type.startswith('t'):
        collect_data=np.hstack([np.abs(result.items_training_errors())[:,dimension] for result in results])
    elif error_type.startswith('p'):
        collect_data=np.hstack([np.abs(result.items_prediction_errors())[:,dimension] for result in results])

    plt.hist(collect_data,**kw,orientation='horizontal',zorder=0)
    if unit_name:
        plt.ylabel('Absolute error %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.ylabel('Absolute error',fontsize=fontsize)
    plt.xlabel('Counts',fontsize=fontsize)
    
    errors=evl.total_items_errors(collect_data)
    if selected_errors==None:
        selected_errors=('RMSE','MAE','50%ile AE','75%ile AE','95%ile AE','MaxAE')
    if marker==None:
        marker={'RMSE':'s','MAE':'x','50%ile AE':'X','75%ile AE':'D','95%ile AE':'+','MaxAE':'.'}
    for selected_error in selected_errors:
        plt.scatter(marker_x,errors[selected_error],s=50,zorder=1,marker=marker[selected_error],label=selected_error)
    plt.legend()
    
def abs_errors_vs_dimension(*results,error_type='traning',unit_name=None,fontsize=20,selected_errors=None,**kw):
    """
    Plot the histogram of absolute errors with box plot for errors
    """
    if error_type.startswith('t'):
        collect_data=np.concatenate([np.abs(result.items_training_errors()) for result in results],axis=0)
    elif error_type.startswith('p'):
        collect_data=np.concatenate([np.abs(result.items_prediction_errors()) for result in results],axis=0)
    
    errors=evl.total_items_errors(collect_data)
    if selected_errors==None:
        selected_errors=('RMSE','MAE','50%ile AE','75%ile AE','95%ile AE','MaxAE')
    for selected_error in selected_errors:
        errors[selected_error].plot(label=selected_error,**kw)
        plt.scatter(errors.index,errors[selected_error])
    
    plt.legend()
    if unit_name:
        plt.ylabel('Absolute error %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.ylabel('Absolute error',fontsize=fontsize)
    plt.xlabel('Dimension of the descriptor',fontsize=fontsize)
    plt.xlim(0,len(errors)+1)
    plt.xticks(range(1,len(errors)+1))
    
def plot2D(result,dimension):
    dimension-=1
    pred=result.training_values()[:,dimension]
    plt.scatter(result.property,pred)
    
    prop_min=result.property.min()
    prop_max=result.property.max()
    x=np.arange(prop_min,prop_max+(prop_max-prop_min)/5,(prop_max-prop_min)/5)
    plt.plot(x,x,c='r')