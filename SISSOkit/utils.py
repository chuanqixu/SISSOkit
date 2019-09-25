import re
from collections import Iterable





class lazyproperty:
    r"""
    Lazy property
    """
    
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value
    
def start_and_number(n_list):
    r"""
    Returns the start index and number in each turn.
    """
    
    n_now=0
    sn=[]
    for i in range(len(n_list)):
        sn.append([n_now,n_list[i]])
        n_now+=n_list[i]
    return sn

def seperate_DataFrame(dataframe,n_list):
    r"""
    Returns the seperated DataFrame.
    """
    
    if isinstance(n_list[0],Iterable)==False:
        return [dataframe.iloc[start:start+n_item]
                for start,n_item in start_and_number(n_list)]
    elif isinstance(n_list[0][0],Iterable)==False:
        n1=[sum(n) for n in n_list]
        data1=seperate_DataFrame(dataframe,n1)
        return [seperate_DataFrame(data,n) for data,n in zip(data1,n_list)]
    
def seperate_list(original_list,n_list):
    r"""
    Returns the seperated list.
    """
    
    if isinstance(n_list[0],Iterable)==False:
        return [original_list[start:start+n_item]
                for start,n_item in start_and_number(n_list)]
    elif isinstance(n_list[0][0],Iterable)==False:
        n1=[sum(n) for n in n_list]
        data1=seperate_DataFrame(original_list,n1)
        return [seperate_DataFrame(data,n) for data,n in zip(data1,n_list)]

def descriptors_to_markdown(expression):
    r"""
    Returns the markdown form of expression
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
                    OPND.append(s)
                s=''
            OPTR.append(expression[i])
            if expression[i]==')':
                OPTR.pop()
                if i+1<len(expression) and expression[i+1]=='^':
                    pattern=re.compile(r'\^-?\d')
                    power=pattern.match(expression,i+1).group()
                    i+=len(power)
                    operand=OPND.pop()
                    if operand.startswith('\\frac'):
                        operand='\\left( %s \\right)'%operand
                    if power=='^-1':
                        OPND.append(operand+'^{-1}')
                    elif power=='^2':
                        OPND.append(operand+'^2')
                    elif power=='^3':
                        OPND.append(operand+'^3')
                    elif power=='^6':
                        OPND.append(operand+'^6')
                    OPTR.pop()
                elif len(OPTR)>1 and OPTR[-1]=='(':
                    operand=OPND.pop()
                    if OPTR[-2]=='exp':
                        OPND.append('e^{%s}'%operand)
                    elif OPTR[-2]=='exp-':
                        OPND.append('e^{- \\left( %s \\right)}'%operand)
                    elif OPTR[-2]=='sqrt':
                        OPND.append('\\sqrt{%s}'%operand)
                    elif OPTR[-2]=='cbrt':
                        OPND.append('\\sqrt[3]{%s}'%operand)
                    elif OPTR[-2]=='log':
                        OPND.append('\\log{\\left( %s \\right)}'%operand)
                    elif OPTR[-2]=='abs':
                        OPND.append('\\left|{%s}\\right|'%operand)
                    elif OPTR[-2]=='sin':
                        OPND.append('\\sin{\\left( %s \\right)}'%operand)
                    elif OPTR[-2]=='cos':
                        OPND.append('\\cos{\\left( %s \\right)}'%operand)
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
                        OPND.append('\\left( %s + %s \\right)'%(operand1,operand2))
                    elif operator=='-':
                        if OPTR[-1]=='abs':
                            OPND.append('\\left|%s - %s\\right|'%(operand1,operand2))
                            OPTR.pop()
                        else:
                            OPND.append('\\left( %s - %s \\right)'%(operand1,operand2))
                    elif operator=='*':
                        OPND.append('\\left( %s * %s \\right)'%(operand1,operand2))
                    elif operator=='/':
                        OPND.append('\\frac{%s}{%s}'%(operand1,operand2))
        i+=1
    return OPND.pop()

def scientific_notation_to_markdown(value):
    r"""
    Returns the markdown form of scientific notation form of value.
    """
    
    value=str(value)
    if 'e' in value or 'E' in value:
        try:
            number,exponent=value.split('e')
        except:
            number,exponent=value.split('E')
        if exponent.startswith('+'):
            exponent=exponent[1:]
            if exponent.startswith('0'):
                exponent=exponent[1:]
        elif exponent[1]=='0':
            exponent='-'+exponent[2:]
        
        if exponent=='0':
            exponent=''
        elif exponent=='1':
            exponent='10'
        else:
            exponent='10^'+'{%s}'%exponent
        return number+' \\times '+exponent
    else:
        return value

def models_to_markdown(regression,task,dimension,indent=''):
    r"""
    Returns the markdown form of models.
    """
    
    coefficients=regression.coefficients[task-1][dimension-1]
    intercepts=regression.intercepts[task-1][dimension-1]
    descriptors=regression.descriptors[dimension-1]
    
    model=scientific_notation_to_markdown(intercepts)
    for d in range(dimension):
        model+=indent
        coeff=scientific_notation_to_markdown(coefficients[d])
        if coeff.startswith('-'):
            model+=' - '
            coeff=coeff[1:]
        else:
            model+=' + '
        model=model+coeff+' \\times '+descriptors_to_markdown(descriptors[d])
    
    return model

