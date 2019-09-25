# SISSOkit

SISSOkit is a Python library for analysis of SISSO, including generating cross validation files, analyzing results, plotting. Data structures of SISSOkit are mainly numpy array, pandas DataFrame or Series and Python built-in data structure like list, so you can easily build your own code based on SISSOkit.

## What is SISSO?

SISSO is short for sure independence screening and sparsifying operator, which is a compressed-sensing method for identifying the best low-dimensional descriptor in an immensity of offered candidates.

References:  
R. Ouyang, S. Curtarolo, E. Ahmetcik, M. Scheffler, and L. M. Ghiringhelli, Phys. Rev. Mater. 2, 083802 (2018).  
R. Ouyang, E. Ahmetcik, C. Carbogno, M. Scheffler, and L. M. Ghiringhelli, J. Phys.: Mater. 2, 024002 (2019).

For SISSO code, please see [SISSO](https://github.com/rouyang2017/SISSO)

## Getting Started

### Dependencies

1. numpy
2. pandas
3. matplotlib

### Installation

Using commend line:
```
pip install SISSOkit
```

### Quick Report

SISSOkit includes some jupyter notebook templates, which you can quickly get a basic analysis of SISSO results without any knowledge about the code. You can find them in the directory `notebook_templates`.

Or you can simply use function in `SISSOkit.notebook`. In this case, you only need to specify path to SISSO results, path to which notebook will generate and the notebook name.

For example:
```python
from SISSOkit import notebook

SISSO_path=[
    'path to SISSO results over whole data set',
    'path to cross validation results'
]
notebook_path='notebook path'
notebook_name='regression with CV'

notebook.generate_report(SISSO_path,notebook_path,notebook_name)
```

Then run all cells, and you will get fundamental analysis of SISSO results.

### Usage

Main idea of SISSOkit is that every SISSO result is an instance. The basic class in SISSOkit is `Regression`, `RegressionCV`, `classification`, `classificationCV` in module `SISSOkit.evaluation`. To instantiate them, you only need to pass directory path to it.

Arguments in `SISSO.in`, descriptors, coefficients and intercepts in `SISSO.out` are all accessible by getting the attributes.

Prediction values, training errors, prediction errors can be acquired by calling the methods.

### Documentation

For more detailed information about SISSOkit, please read documentation `./docs/build/html/index.html`, which is generated automatically by sphinx.