import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

def appEq(a,b):
    if abs(a-b)<a*0.005:
        return True
    else: return False

def trend(ema,i,x):
    f = lambda b: int(ema[b]>ema[b-1])
    k = 0
    while(f(i-k)==x): k+=1
    return k

file_name = "stock.csv"
data = pd.read_csv(file_name)
row_name = "High"
arr = np.array(data[row_name],dtype=float)

ema = np.copy(arr) ; weight = 0.01
N = len(ema)
for i in range(1,N):
    ema[i] = ema[i-1] + weight*(arr[i]-ema[i-1])

net_wealth = [] ; share_num = 1
for i in range(2,N):
    if appEq(ema[i],arr[i]):
        share_num = trend(ema,i,1)
        if share_num!=0:
            net_wealth.append(arr[i]*share_num) #sell
        else:
            share_num = trend(ema,i,0)
            net_wealth.append(-arr[i]*share_num) #buy

money = 0 ; wealth_graph = []
n = len(net_wealth)
wealth_graph.append(net_wealth[0]+net_wealth[1])
for i in range(2,n-n%2,2):
    temp = net_wealth[i]+net_wealth[i+1]
    wealth_graph.append(wealth_graph[i//2-1]+net_wealth[i]+net_wealth[i+1])

gross_prof,gross_loss = 0,0
for item in net_wealth:
    if item>=0: gross_prof+=item
    else: gross_loss-=item
prof_factor = gross_prof/gross_loss
print("Net profit: ",round(wealth_graph[-1],2))
print("Profit factor: ",round(prof_factor,2))

pl.plot(arr,color='green') ; pl.plot(ema,color='red') 
pl.xlabel("Days") ; pl.ylabel("Price")
pl.title("Trading strategy using EMA")
pl.legend([" Stock price","Exponential Moving Average"])
pl.show()
pl.plot(np.array(wealth_graph))
pl.xlabel("Days") ; pl.ylabel("Net profit")
pl.title("Total profit")
pl.show()
