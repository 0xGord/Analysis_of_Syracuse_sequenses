import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import log,sqrt,exp
import time
from sys import exit
import random as rnd
def kollats_hyp(n:int):
    if(n == 0):
        exit()
    number    = []
    iteration = []
    
    counter = 0
    number.append(n)
    iteration.append(counter)
    
    while (n != 1):
        if n % 2 == 0:
            n /= 2
        else:
            n = 3 * n + 1
        counter += 1

        number.append(int(n))
        iteration.append(counter)
    return (iteration, number)
                 
def syracuse_sequences (n:int):
    counter = 0
    number = []
    iterations = []
    for i in range(1, n):
        number.append(i)
        while (i != 1):
            if i % 2 == 0:
                i /= 2
            else:
                i = 3 * i + 1   
            counter += 1
        iterations.append(counter)
        counter = 0
    return (number, iterations)

def distribution (iter_list:list):
    
    iteration = []
    occur     = []
    
    for i in range(max(iter_list)):
        iteration.append(i)
        occur.append(iter_list.count(i)/len(iter_list))
        #occur.append(iter_list.count(i))
        
    return (iteration, occur)

def cdf (iter_list:list):
    
    val = []
    temp = 0
    for i in iter_list:
        temp += i
        val.append(temp)
    return val

def draw_graph(input_list:list):
    G = nx.Graph()

    nodes = []
    edges = []
    
    temp = kollats_hyp(input_list[0])[1]
    nodes += temp
    
    for j in range(len(temp)-1):
        edges.append((temp[j], temp[j+1]))
        
    for i in range(1, len(input_list)):      
        j = 1
        temp = []
        temp = kollats_hyp(input_list[i])[1]
        for j in temp:
            if (j not in nodes):
                nodes.append(j)

        for j in range(len(temp)-1):
            edges.append((temp[j], temp[j+1]))

    edges.append((1,4))
    
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    nx.draw(G, with_labels=True, node_color = 'lime', font_weight='bold')
    plt.show()             

def max_friequency():
    N = [c**2*100 for c in range(1,30)]

    f = []
    for n in N:
        f.append(max(distribution(syracuse_sequences(n)[1])[1]))
    return (N, f)

def means():
    N = [c**2*1000 for c in range(1,10)]
    #N = [c for c in range(5,1000)]
    #N = [round(2**c)*100 for c in range(1,13)]
    m = []
    v = []
    for n in N:
        val = syracuse_sequences(n)[1]
        m.append(np.mean(val))
        v.append(np.var(val))
    #print(*zip(N,m))
    #print((N,m))
    return (N, m, v)
    
def log_regression(x:list, y:list):
    sigma1 = 0
    sigma2 = 0
    sigma3 = 0
    sy = sum(y)
    
    for i in range(len(x)):
        sigma1 += y[i]*log(x[i])
        sigma2 += log(x[i])
        sigma3 += log(x[i])**2
    a = (len(x)*sigma1 - sigma2*sy)/(len(x)*sigma3-sigma2**2)
    b = 1/len(x)*sy - a/len(x)*sigma2
    return a, b


def kollats_pseudorand(n:int):
    res = kollats_hyp(n)[0][-1] + 1
    return res

def main():
    N=2**13-1
    res = kollats_hyp(N)
    fig = plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.plot(res[0], res[1], '-o', color = "green")
    plt.vlines(res[0][res[1].index(max(res[1]))],0, max(res[1]), color = "red", linestyle = "dashed")
    plt.text(res[0][res[1].index(max(res[1]))]+1/2, max(res[1]), f"({res[0][res[1].index(max(res[1]))]}, {int(max(res[1]))}) max",)
    plt.xlabel("Iteration")
    plt.ylabel("Number")
    plt.title(f"Syracuse sequense starting from {N}")
    plt.show()
    #print(max(res[0]))
    print(kollats_hyp(10))
    N=10000
    res = syracuse_sequences(N)
    fig = plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.plot(res[0], res[1], '.')
    plt.xlabel("Number")
    plt.ylabel("Lenght")
    plt.title(f"Lenght of Syracuse sequences for numbers from 1 to {N}")
    x = np.linspace(0.1, N, num = N)
    y = []
    for i in list(x):
        y.append(log(i, 2))
    plt.plot(x,y, color='red', label = u'm = log\u2082(n)')

    x = np.linspace(0.1, N, num = N)
    y = []
    for i in list(x):
        y.append((log(i)+1)*(1/log(2)+10.4) - 1/log(2)-11.8)
    plt.plot(x,y, color='m')
    y = []
    for i in list(x):
        y.append(10.4*log(i) - 11.8)
    plt.plot(x,y)
    #fig = plt.figure(figsize=(8,5))
    #plt.grid(True)
    #plt.plot(means()[0], means()[1], label = "Mean")
    plt.legend()
    plt.show()
    
    res_ = distribution(res[1])
    fig = plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.vlines(res_[0],0, res_[1])
    plt.plot(res_[0], res_[1], '.', color = "purple")
    plt.xlabel("Lenght")
    plt.ylabel("Relative frequency")
    plt.title(f"Distribution of lengths of Syracuse sequences for numbers from 1 to {N}")
    plt.show()
    
    fig = plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.hist(res[1], round(1+3.2*log(len(res_[0]))), density='True', alpha = 0.5, edgecolor = 'blue')
    #plt.axvline(np.mean([res[1]]))
    plt.xlabel("Lenght")
    plt.ylabel("Relative frequency")
    plt.title(f"Histogram of lengths of Syracuse sequences for numbers from 1 to {N}")
    plt.show()
    
    fig = plt.figure(figsize=(8,5))
    draw_graph([128,3,7])
    plt.show()

    fig = plt.figure(figsize=(5,5))
    plt.grid(True)
    x=[]
    y=[]
    S = 10000
    for i in range(1,S):
        #x.append(kollats_pseudorand(rnd.randint(1, N)))
        x.append(i)
        y.append(kollats_pseudorand(rnd.randint(1,N)))
    plt.plot(x, y, ',')
    plt.show()
    plt.hist(y,round(1+3.2*log(len(y))), density='True', alpha = 0.5, edgecolor = 'blue')
    plt.show()
    
    fig = plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.plot(means()[0], means()[1], label = "Mean")
    coef = log_regression(means()[0], means()[1])
    print(coef)
    y = []
    for i in list(means()[0]):
        y.append(coef[0]*log(i) + coef[1])
    plt.plot(means()[0], y, 'x', label = f'{coef[0]:.2f} ln(N) + ({coef[1]:.2f})')
    #plt.plot(means()[0], means()[2], label = "Var")
    plt.legend()
    plt.xlabel("Sample size")
    plt.show()
    print('Number:', means()[0], '\n', 'Mean:', means()[1])
    
    
if __name__ =="__main__":
    main()
