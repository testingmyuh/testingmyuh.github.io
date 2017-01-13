---
layout: post
title: Combination of CP measurements
tags:
- News
- Tags
- jupyter
- Post
---


## init stuff


```python
import numpy as np
from numpy.linalg import inv
import scipy.optimize as opt

%matplotlib inline
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rc_file('/Users/pietro/GoogleDrive/PythonScripts/matplotlibrc')
```


```python
#Guassian pdfs
def gauss()
```

# Main part

## Input measurements


```python
'''
    measurement dictionary.
    measurement as: ['which variable measure', (xxx, sigma_stat, sigma_syst)]
'''
dmeas = {'A_Gamma SL' : ['-Aind', (-1.25e-3, 0.73e-3, 0)],
         'A_Gamma prompt KK': ['-Aind', (-0.35e-3, 0.62e-3, 0.12e-3)],
         'A_Gamma prompt PP': ['-Aind', (0.33e-3, 1.06e-3, 0.14e-3)],
         'DeltaACP SL' : ['DAdir+Aind', (1.4e-3, 1.6e-3, 0.8e-3)],
         'DeltaACP D*' : ['DAdir+Aind', (-1.0e-3, 0.8e-3, 0.3e-3)]
         }
# delta t/tau, <t>/tau
times = {
    'DeltaACP SL' : [ (0.014, 0.04, 0), (1.051, 0.001, 0.004) ],
    'DeltaACP D*' : [ (0.1153, 0.0007, 0.0018), (2.0949, 0.0004, 0.0159)]
}

myCP = (5.5e-3, 6.3e-3, 4.1e-3)

colors = { 
    'A_Gamma SL': 'green',
    'A_Gamma prompt KK' : 'red',
    'A_Gamma prompt PP' : 'blue',
    'DeltaACP SL' : 'pink',
    'DeltaACP D*': 'blue'
    }
```


## Calc Chi2 and minimize 


```python
def chisq_backup( (x0, y0) ):
    '''
        chisq function of a two dimensional function
    '''
    model = np.array((x0, y0))
    chisq = 0
    for m in dobs.iteritems():
        #print 'adding measurements +++', m[0], '+++'
        #(a-b).transpose().dot(mm).dot(a-b)
        #print m
        delta_m = (m[1][0]-model)
        cov_inv = inv(m[1][1])
        chisq += delta_m.transpose().dot(cov_inv).dot(delta_m)
    return float(chisq)

def chisq( (Aind, DAdir, yCP, deltat_SL, avgt_SL, deltat_Dst, avgt_Dst) ):
    '''
        chisq function of a two dimensional function
    '''
    chisq = 0
    for name, val in dmeas.iteritems():
        meas = val[1]
        if val[0] == '-Aind':
            model = -Aind
        elif val[0] == 'DAdir+Aind':
            model = DAdir*(1 + times[name][1][0]*myCP[0]) + times[name][0][0]*Aind
        sigma = np.sqrt( meas[1]**2 + meas[2]**2 )
        delta = (meas[0]-model)
        chisq += (delta/sigma)**2
        
    for name, val in times.iteritems():
        if 'SL' in name:
            sigma = np.sqrt( times[name][0][1]**2 + times[name][0][2]**2 )
            chisq += ((times[name][0][0]-deltat_SL)/sigma)**2
        
            sigma = np.sqrt( times[name][1][1]**2 + times[name][1][2]**2 )
            chisq += ((times[name][1][0]-avgt_SL)/sigma)**2
        else:
            sigma = np.sqrt( times[name][0][1]**2 + times[name][0][2]**2 )
            chisq += ((times[name][0][0]-deltat_Dst)/sigma)**2
        
            sigma = np.sqrt( times[name][1][1]**2 + times[name][1][2]**2 )
            chisq += ((times[name][1][0]-avgt_Dst)/sigma)**2
        
    sigma = np.sqrt( myCP[1]**2 + myCP[2]**2 )
    chisq += ((myCP[0]-yCP)/sigma)**2
        
    return chisq
```


```python
res = opt.minimize(chisq, (0.3,  0.2, 0, 0.01, 1, 0.1, 2), method ='Nelder-Mead')
```


```python
res
```




     final_simplex: (array([[  5.44852914e-04,  -6.01152119e-04,   1.17522624e-03,
              1.64230431e-02,   1.05098840e+00,   1.15289340e-01,
              2.09490243e+00],
           [  5.46844787e-04,  -6.02373898e-04,   1.17515296e-03,
              1.64223907e-02,   1.05098994e+00,   1.15296704e-01,
              2.09500037e+00],
           [  5.45978981e-04,  -6.02446772e-04,   1.17504186e-03,
              1.64229796e-02,   1.05097840e+00,   1.15302362e-01,
              2.09484805e+00],
           [  5.44030614e-04,  -6.00665589e-04,   1.17507585e-03,
              1.64223433e-02,   1.05098846e+00,   1.15307494e-01,
              2.09495397e+00],
           [  5.43302720e-04,  -6.04578944e-04,   1.17528494e-03,
              1.64232683e-02,   1.05099717e+00,   1.15287494e-01,
              2.09486782e+00],
           [  5.44637585e-04,  -5.96128094e-04,   1.17514573e-03,
              1.64226000e-02,   1.05100452e+00,   1.15300953e-01,
              2.09488805e+00],
           [  5.48733662e-04,  -6.05228663e-04,   1.17507864e-03,
              1.64221764e-02,   1.05100755e+00,   1.15311301e-01,
              2.09493684e+00],
           [  5.43383447e-04,  -6.01767665e-04,   1.17525931e-03,
              1.64228315e-02,   1.05103604e+00,   1.15296551e-01,
              2.09484651e+00]]), array([ 3.56204303,  3.56205429,  3.56206091,  3.56207648,  3.56208012,
            3.56208812,  3.56210177,  3.56211762]))
               fun: 3.5620430307584825
           message: 'Optimization terminated successfully.'
              nfev: 477
               nit: 300
            status: 0
           success: True
                 x: array([  5.44852914e-04,  -6.01152119e-04,   1.17522624e-03,
             1.64230431e-02,   1.05098840e+00,   1.15289340e-01,
             2.09490243e+00])




```python
#error
print 2*res.x
chisq(1e-1000*res.x)
```

    [  1.08970583e-03  -1.20230424e-03   2.35045248e-03   3.28460862e-02
       2.10197680e+00   2.30578680e-01   4.18980486e+00]





    85894.929296938179



## plot measurements


```python
x = np.linspace(-10e-3, 10e-3, 200)
y = np.linspace(-10e-3, 10e-3, 200)
#X, Y = np.meshgrid(x, y)
```


```python
minpoint = res.x[2:]
mypoint =  np.zeros(len(res.x))
mypoint[2:] = minpoint
z = np.zeros( (len(x), len(y)) )
for i, myy in enumerate(y):
    for j, myx in enumerate(x):
        mypoint[0] = myx
        mypoint[1] = myy
        z[i][j] = chisq( mypoint )
```


```python
z[np.abs(z-(res.fun+1))<1e-1]
```




    array([ 4.60851656,  4.53741013,  4.57290822,  4.53679779,  4.47884666,
            4.59975161,  4.53272323,  4.64541805,  4.60910908,  4.52815892,
            4.51838105,  4.46235387,  4.50247737,  4.46237592,  4.58525356,
            4.55332185,  4.49150927,  4.64336248,  4.56648909,  4.560405  ,
            4.53904438,  4.62428828])




```python
CS = plt.contour(x,y, z,levels=[res.fun+1, res.fun+2.30, res.fun+6.17, res.fun+11.8])
plt.clabel(CS, inline=1, fontsize=10)
#plt.contour(x,y, z, level=[85500, 85900])
```




    <a list of 3 text.Text objects>




![png]({{ site.baseurl }}static/img/output_16_1.png)



```python

```
