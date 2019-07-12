n = 11


# https://stackoverflow.com/questions/26560726/python-binomial-coefficient
from math import factorial as fac
from math import log

def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


data = [(str(bin(x))[2:]) for x in range(2 ** n)]
data = ["0"*(n-len(x)) + x for x in data]
#print(data)

majority = set([x for x in data if len([y for y in x if y == '1']) > n/2.0])


def binaryEntropy(p):
    if p == 0 or p == 1:
        return 0
    return - ( p * log(p) + (1-p) * log(1-p))

total = 0
entropies = []
for i in range(0, n+1):
   ps = 0
   entropy = 0
   for j in range(0, i+1):
     prob = binomial(i, j) / float(2**i)
     #print(prob)
     probe = ("0"*j) + ("1"*(i-j))
     assert len(probe) == i
#     print(data)
     starts = [x for x in data if x.startswith(probe)]
     good = len([x for x in starts if x in majority])
#     print(probe)
     assert len(starts) > 0
     expSurp = binaryEntropy(good / float(len(starts)))
#     print(i, j, expSurp)
     ps += prob
     entropy += prob * expSurp
   assert ps == 1, ps
#   marginalEntropy = 

   entropies.append( (entropy))
   resultHere = 0
   # the set contains i variables + the result

   # subsets of j(<=i) variables + the result
   for j in range(0, i+1):
       resultHere -= binomial(i,j) * (entropies[j] + log(2) * j) * (-1)**(j+1)
   # subsets of j(<=i) variables
   for j in range(0, i+1):
       resultHere -= binomial(i,j) * entropies[j] * (-1)**(j)
   print(i, entropy, resultHere)
   



