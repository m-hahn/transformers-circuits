# WORKS confirmed by manual calculation

n = 3


# https://stackoverflow.com/questions/26560726/python-binomial-coefficient
from math import factorial as fac
from math import log

def binomial(x, y):
    if y > x:
        return 0
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


data = [(str(bin(x))[2:]) for x in range(2 ** n)]
data = ["0"*(n-len(x)) + x for x in data]
dataM = [("1" if (len([y for y in x if y == '1']) > n/2.0) else "0") + x for x in data]



def binaryEntropy(p):
    if p == 0 or p == 1:
        return 0
    return - ( p * log(p) + (1-p) * log(1-p))

total = 0
entropies = [None for _ in range(n+2)]
for i in range(1, n+2):
   subsetsWithResponse = [x[:i] for x in dataM]
   counts = {}
   for x in subsetsWithResponse:
     counts[x] = counts.get(x,0)+1
   entropy = - sum([float(y)/len(subsetsWithResponse) * (log(y) - log(len(subsetsWithResponse))) for x, y in counts.iteritems()])
   entropies[i] = entropy

print(entropies)

for i in range(1, n+2):
    result = 0
    for j in range(1, i+1):
        # subsets of size j, where one of them is the response
        result += (-1)**j * binomial(i-1, j-1) * entropies[j]
        # subsets of size j, where none of them is the response
        result += (-1)**j * binomial(i-1, j) * log(2) * j
    print(i, result)



