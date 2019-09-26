from . import evaluation as evl
from . import cross_validation as cv

import numpy as np
import matplotlib.pyplot as plt
import functools





def baselineplot(regression,unit=None,marker='x',marker_color='r',marker_y=1,marker_shape=100,fontsize=20,**kw):
    r"""
        Plots the histgram of baseline of regression.
    
        Arguments:
            regression (evaluation.Regression): 
                regression result.
            
            unit (string): 
                unit of property.
            
            marker (char): 
                marker type of mean value. It's the same as in matplotlib.
            
            marker_color (char): 
                marker color of mean value. It's the same as in matplotlib.
            
            marker_shape (int): 
                marker shape of mean value. It's the same as in matplotlib.
            
            marker_y (float): 
                vertical coordinate of marker of mean value.
            
            fontsize (int): 
                fontsize of axis name.
            
            kw: 
                keyword arguments of histogram. It's the same as ``hist()`` in matplotlib.
    """

    plt.hist(regression.property,zorder=0,**kw)
    plt.scatter(regression.property.mean(),marker_y,marker=marker,c=marker_color,s=marker_shape,zorder=1,label='Mean Value')
    if unit:
        plt.xlabel(regression.property_name+'['+unit+']',fontsize=fontsize)
    else:
        plt.xlabel(regression.property_name,fontsize=fontsize)
    plt.ylabel('Counts',fontsize=fontsize)
    plt.title('Histogram of %s over training data\nMean = %.5f\nStandard deviation = %.5f'%(regression.property_name,regression.property.mean(),regression.property.std()),fontsize=fontsize)
    plt.legend()



def error_hist(dimension,*regressions,training=True,absolute=False,unit=None,fontsize=20,**kw):
    r"""
        Plots the histgram of errors of regression.
    
        Arguments:
            dimension (int): 
                dimension of descriptor.
            
            regressions (evaluation.Regression or evaluation.RegressionCV): 
                regression result.
            
            training (bool): 
                training errors or prediction errors.
            
            absolute (bool): 
                absolute errors or not.
            
            unit (string): 
                unit of property.
            
            fontsize (int): 
                fontsize of axis name.
            
            kw: 
                keyword arguments of histogram. It's the same as ``hist()`` in matplotlib
    """
    
    dimension-=1
    if absolute:
        collect_data=np.hstack([np.abs(regression.errors(training=training))[dimension,:] for regression in regressions])
        if unit:
            plt.xlabel('Absolute error %s'%('['+unit+']'),fontsize=fontsize)
        else:
            plt.xlabel('Absolute error',fontsize=fontsize)
    else:
        collect_data=np.hstack([regression.errors(training=training)[dimension,:] for regression in regressions])
        if unit:
            plt.xlabel('Signed error %s'%('['+unit+']'),fontsize=fontsize)
        else:
            plt.xlabel('Signed error',fontsize=fontsize)
    plt.hist(collect_data,**kw)
    plt.ylabel('Counts',fontsize=fontsize)



def prediction_vs_property(dimension,*regressions,training=True,unit=None,fontsize=20,**kw):
    r"""
        Plots the scatter plot of prediction v.s. property.
    
        Arguments:
            dimension (int): 
                dimension of descriptor.

            regressions (evaluation.Regression or evaluation.RegressionCV): 
                regression result.
            
            training (bool): 
                training errors or prediction errors.
            
            unit (string): 
                unit of property.
            
            fontsize (int): 
                fontsize of axis name.
            
            kw: 
                keyword arguments of histogram. It's the same as ``hist()`` in matplotlib
    """
    
    dimension-=1
    if isinstance(regressions[0],evl.RegressionCV):
        if training:
            property_values=np.hstack([regression.property[cv].values for regression in regressions for cv in range(0,regression.n_cv)])
        else:
            property_values=np.hstack([regression.validation_data[cv].iloc[:,1].values.tolist() for regression in regressions for cv in range(0,regression.n_cv)])
        prediction_value=np.hstack([regression.predictions(training=training)[dimension,:] for regression in regressions])-property_values
    else:
        property_values=np.hstack([regression.property.values for regression in regressions])
        prediction_value=np.hstack([regression.predictions(training=training)[dimension,:] for regression in regressions])-property_values
    plt.scatter(property_values,prediction_value,**kw)
    if unit:
        plt.xlabel('%s in data set %s'%(regressions[0].property_name,'['+unit+']'),fontsize=fontsize)
        plt.ylabel('Signed error %s'%('['+unit+']'),fontsize=fontsize)
    else:
        plt.xlabel('%s in data set'%(regressions[0].property_name),fontsize=fontsize)
        plt.ylabel('Signed error',fontsize=fontsize)



def hist_with_markers(dimension,*regressions,training=True,unit=None,fontsize=20,selected_errors=None,marker_x=0,marker=None,**kw):
    r"""
        Plots the histogram of absolute errors with markers
    
        Arguments:
            dimension (int): 
                dimension of descriptor.
            
            regressions (evaluation.Regression or evaluation.RegressionCV): 
                regression result.
            
            training (bool): 
                training errors or prediction errors.
            
            unit (string): 
                unit of property.
            
            fontsize (int): 
                fontsize of axis name.
            
            seleted_errors (None or list): 
                what errors should pinpoint in the plot.
                errors are 'RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'.
                If it is ``None``, then all the errors will appear in the plot.
            
            marker_x (float): 
                horrizontal coordinate of marker.
            
            marker (NOne or list): 
                marker type. It's the same as in matplotlib.
                If it is ``None``, then will use the default types.
            
            kw: 
                keyword arguments of histogram. It's the same as ``hist()`` in matplotlib
    """

    dimension-=1
    collect_data=np.hstack([np.abs(regression.errors(training=training))[dimension,:] for regression in regressions])
    plt.hist(collect_data,**kw,orientation='horizontal',zorder=0)
    if unit:
        plt.ylabel('Absolute error %s'%('['+unit+']'),fontsize=fontsize)
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



def abs_errors_vs_dimension(*regressions,training=True,unit=None,fontsize=20,selected_errors=None,display_baseline=False,label='',**kw):
    r"""
        Plots the histogram of absolute errors with box plot for errors.
    
        Arguments:
            regressions (evaluation.Regression or evaluation.RegressionCV): 
                regression result.
            
            training (bool): 
                training errors or prediction errors.
            
            unit (string): 
                unit of property.
            
            fontsize (int): 
                fontsize of axis name.
            
            seleted_errors (None or list): 
                what errors should appear in the plot.
                errors are 'RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE'.
                If it is ``None``, then all the errors will appear in the plot.
            
            display_baseline (bool): 
                whether plot baseline.
            
            kw: 
                keyword arguments of histogram. It's the same as ``hist()`` in matplotlib
    """

    if display_baseline:
        plt.plot([1,regressions[0].dimension],[regressions[0].baseline[1],regressions[0].baseline[1]],'--',label='Baseline')
    collect_data=np.hstack([np.abs(regression.errors(training=training)) for regression in regressions])
    errors=evl.compute_errors(collect_data)
    if selected_errors==None:
        selected_errors=('RMSE','MAE','25%ile AE','50%ile AE','75%ile AE','95%ile AE','MaxAE')
    for selected_error in selected_errors:
        errors[selected_error].plot(label=label+' '+selected_error,**kw)
        plt.scatter(errors.index.values,errors[selected_error].values)
    
    plt.legend()
    if unit:
        plt.ylabel('Errors %s'%('['+unit+']'),fontsize=fontsize)
    else:
        plt.ylabel('Errors',fontsize=fontsize)
    plt.xlabel('Dimension of the descriptor',fontsize=fontsize)
    plt.xlim(0,len(errors)+1)
    plt.xticks(range(1,len(errors)+1))



def boxplot(regression,training=True,unit=None,fontsize=20,**kwargs):
    r"""
        Plots the boxplot of regression.
    
        Arguments:
            regression (evaluation.Regression or evaluation.RegressionCV): 
                regression result.
            
            training (bool): 
                training errors or prediction errors.
            
            unit (string): 
                unit of property.
            
            fontsize (int): 
                fontsize of axis name.
            
            kw: 
                keyword arguments of histogram. It's the same as ``hist()`` in matplotlib
    """
    
    plt.boxplot([np.abs(regression.errors(training=training))[dimension] for dimension in range(regression.dimension)],
                **kwargs)
    if unit:
        plt.ylabel('Errors %s'%('['+unit+']'),fontsize=fontsize)
    else:
        plt.ylabel('Errors',fontsize=fontsize)
    plt.xlabel('Dimension',fontsize=fontsize)



def errors_details(regression,training=True):
    r"""
        Plots the detailed information about regression, including histograme of signed errors,
        preditction v.s. property and histgram of absolute errors with markers.
    
        Arguments:
            regression (evaluation.Regression or evaluation.RegressionCV): 
                regression result.
            
            training (bool): 
                training errors or prediction errors.
    """
    
    plt.figure(figsize=(20,6*regression.dimension))
    for i in range(1,regression.dimension+1):
        plt.subplot(regression.dimension,3,(i-1)*3+1)
        error_hist(i,regression,absolute=False,training=training,rwidth=0.8)
        plt.subplot(regression.dimension,3,(i-1)*3+2)
        prediction_vs_property(i,regression,training=training)
        plt.subplot(regression.dimension,3,(i-1)*3+3)
        hist_with_markers(i,regression,training=training, bins=20, alpha=0.5, rwidth=0.8,marker_x=10)
