import matplotlib.pyplot as plt
import SISSO_analysis.evaluation as evl
import SISSO_analysis.cross_validation as cv
import SISSO_analysis.plot as plot
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

def error_hist(dimension,*results,training=True,abs=False,unit_name=None,fontsize=20,**kw):
    """
    Plot the histogram of signed or absolute error
    """
    dimension-=1
    if abs:
        collect_data=np.hstack([np.abs(result.errors(training=training))[dimension,:] for result in results])
        if unit_name:
            plt.xlabel('Signed error %s'%('['+unit_name+']'),fontsize=fontsize)
        else:
            plt.xlabel('Signed error',fontsize=fontsize)
    else:
        collect_data=np.hstack([result.errors(training=training)[dimension,:] for result in results])
        if unit_name:
            plt.xlabel('Absolute error %s'%('['+unit_name+']'),fontsize=fontsize)
        else:
            plt.xlabel('Absolute error',fontsize=fontsize)
    plt.hist(collect_data,**kw)
    plt.ylabel('Counts',fontsize=fontsize)
    
def property_vs_prediction(dimension,*results,training=True,unit_name=None,fontsize=20,**kw):
    """
    Plot the scatter plot of property_vs_prediction
    """
    dimension-=1
    if isinstance(results[0],evl.Results):
        if training:
            property_values=np.hstack([result.property[cv].values for result in results for cv in range(0,result.cv_number)])
        else:
            property_values=np.hstack([result.validation_data[cv].iloc[:,1].values.tolist() for result in results for cv in range(0,result.cv_number)])
        prediction_value=np.hstack([result.predictions(training=training)[dimension,:] for result in results])-property_values
    else:
        property_values=np.hstack([result.property.values for result in results])
        prediction_value=np.hstack([result.predictions(training=training)[dimension,:] for result in results])-property_values
    plt.scatter(property_values,prediction_value,**kw)
    if unit_name:
        plt.xlabel('%s in data set %s'%(results[0].property_name,'['+unit_name+']'),fontsize=fontsize)
        plt.ylabel('Signed error %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.xlabel('%s in data set'%(results[0].property_name),fontsize=fontsize)
        plt.ylabel('Signed error',fontsize=fontsize)
        
def hist_and_box_plot(dimension,*results,training=True,unit_name=None,fontsize=20,selected_errors=None,marker_x=0,marker=None,**kw):
    """
    Plot the histogram of absolute errors with box plot for errors
    """
    dimension-=1
    collect_data=np.hstack([np.abs(result.errors(training=training))[dimension,:] for result in results])
    plt.hist(collect_data,**kw,orientation='horizontal',zorder=0)
    if unit_name:
        plt.ylabel('Absolute error %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.ylabel('Absolute error',fontsize=fontsize)
    plt.xlabel('Counts',fontsize=fontsize)
    
    errors=evl.compute_errors(collect_data)
    if selected_errors==None:
        selected_errors=('RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE')
    if marker==None:
        marker={'RMSE':'s','MAE':'x','25%ile AE':'p','50%ile AE':'X','75%ile AE':'D','95%ile AE':'+','MaxAE':'.'}
    for selected_error in selected_errors:
        plt.scatter(marker_x,errors[selected_error],s=50,zorder=1,marker=marker[selected_error],label=selected_error)
    plt.legend()
    
def abs_errors_vs_dimension(*results,training=True,unit_name=None,fontsize=20,selected_errors=None,label='',**kw):
    """
    Plot the histogram of absolute errors with box plot for errors
    """
    collect_data=np.hstack([np.abs(result.errors(training=training)) for result in results])
    errors=evl.compute_errors(collect_data)
    if selected_errors==None:
        selected_errors=('RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE')
    for selected_error in selected_errors:
        errors[selected_error].plot(label=label+':'+selected_error,**kw)
        plt.scatter(errors.index.values,errors[selected_error].values)
    
    plt.legend()
    if unit_name:
        plt.ylabel('Errors %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.ylabel('Errors',fontsize=fontsize)
    plt.xlabel('Dimension of the descriptor',fontsize=fontsize)
    plt.xlim(0,len(errors)+1)
    plt.xticks(range(1,len(errors)+1))

def boxplot(result,training=True,unit_name=None,fontsize=20,**kwargs):
    plt.boxplot([np.abs(result.errors(training=training))[dimension] for dimension in range(result.dimension)],
                **kwargs)
    if unit_name:
        plt.ylabel('Errors %s'%('['+unit_name+']'),fontsize=fontsize)
    else:
        plt.ylabel('Errors',fontsize=fontsize)
    plt.xlabel('Dimension',fontsize=fontsize)