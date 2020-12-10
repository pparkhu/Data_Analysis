import random
import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import ticker, cm
from magpylib._lib.mathLib import fastSum3D

'''
n = 36
x = np.arange(0,n+1,1)
print('x: ', x.shape)

def factorial_for(n):
    ret = 1
    for i in range(1, n+1):
        ret *= i
    return ret

p = 1/2
q = 1/2
N = factorial_for(n)
result = []
result1 = []

xx = 23
binominal = (p **(xx)) * (q **(n-xx)) * N / factorial_for(int(xx)) / factorial_for(int(n-xx))
print("binominal: ", binominal)
average = n*p
sigma = np.sqrt(n*p*q)
gaussian = 1/sigma/np.sqrt(2*np.pi)*np.exp(-(xx-average)**2/2/sigma**2)
print("gaussian: ", gaussian)

for i in range(0,n+1):
    pp = (p ** (x[i])) * (q ** (n - x[i])) * N / factorial_for(int(x[i])) / factorial_for(int(n - x[i]))
    result.append(pp)

for i in range(0,n+1):
    pp = 1/sigma/np.sqrt(2*np.pi)*np.exp(-(x[i]-average)**2/2/sigma**2)
    result1.append(pp)

plt.figure(figsize=(16, 8))
plt.subplot(1,2,1)
#plt.plot(x,result)
plt.scatter(x,result)
#plt.xlim(0,10)
#plt.ylim(-5,5)
#plt.xticks(np.arange(0.,10.01,1))
#plt.yticks(np.arange(-30.00,0.01,10))
plt.title("Binominal ditribution")
plt.xlabel("X")
plt.ylabel("Probability of X")

plt.subplot(1,2,2)
#plt.plot(x,result)
plt.scatter(x,result1)
#plt.xlim(0,10)
#plt.ylim(-5,5)
#plt.xticks(np.arange(0.,10.01,1))
#plt.yticks(np.arange(-30.00,0.01,10))
plt.title("Gaussian ditribution")
plt.xlabel("X")
plt.ylabel("Probability of X")

plt.show()
'''

#--------------------------------------------------------------#

'''
#Poisson Distribution

def factorial_for(n):
    ret = 1
    for i in range(1, n+1):
        ret *= i
    return ret

nn = 71
#x = np.arange(0,nn+1,1)
#x = list(x)


x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
     37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]
m = 64
pp = []

for i in range(0,len(x)):
    p = m**x[i]/factorial_for(x[i])/np.exp(m)
    pp.append(p*float(100))

#print(pp)
print(100-sum(pp))
'''

#--------------------------------------------------------------#

'''
n = [5,10,20,30]
a = 3

x = np.arange(0,n[a]+1,1)
print('x: ', x.shape)
print(x)

def factorial_for(n):
    ret = 1
    for i in range(1, n+1):
        ret *= i
    return ret

p = 0.6
q = 0.4
N = factorial_for(n[a])
result = []

for i in range(0,n[a]+1):
    binominal = (p **(x[i])) * (q **(n[a]-x[i])) * N / factorial_for(int(x[i])) / factorial_for(int(n[a]-x[i]))
    result.append(binominal)

plt.plot(x,result)
plt.scatter(x,result)
plt.show()
'''

#--------------------------------------------------------------#

rand = []
x = np.linspace(0,1,1000)
rand2 = []

for i in range(0, 1000):
    for j in range(0,1000):
        ran = random.random() + 0*x[j]
        rand.append(ran)
    rand2.append(sum(rand)/1000+i*0)
    ran = []
    rand = []

plt.hist(rand2, bins=100, density = 1.0)
plt.show()
