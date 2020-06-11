# WORKS confirmed by manual calculation
# NOTE only works for symmetric functions

n = 18


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
dataM = [("1" if (len([y for y in x if y == '1']) % 4 == 0) else "0") + x for x in data]



def binaryEntropy(p):
    if p == 0 or p == 1:
        return 0
    return - ( p * log(p) + (1-p) * log(1-p))

total = 0
entropies = [None for _ in range(n+2)]
fourier_coefs = [None for _ in range(n+2)]
for i in range(1, n+2):
   print(i)
   subsetsWithResponse = [x[:i] for x in dataM]
   counts = {}
   for x in subsetsWithResponse:
     counts[x] = counts.get(x,0)+1
   entropy = - sum([float(y)/len(subsetsWithResponse) * (log(y) - log(len(subsetsWithResponse))) for x, y in counts.iteritems()])
   fourier_coefs[i] = sum([y * (1 if len([z for z in x if z == "1"]) % 2 == 0 else -1) for x, y in counts.iteritems()]) / float(len(subsetsWithResponse))
   entropies[i] = entropy

print(entropies)
sumISoFar = 0
sumSoFar = 0
synergySoFar = 0
fourier_mass_total = 0
sensitivity = 0
print(fourier_coefs)
print("\t".join([str(x) for x in ["i", "res", "her", "sumSF", "sumISF", "synSF", "fc[i]", "fm[i]", "fm", "sens"]]))

for i in range(1, n+2):
    result = 0
    for j in range(1, i+1):
        # subsets of size j, where one of them is the response
        result += (-1)**j * binomial(i-1, j-1) * entropies[j]
        # subsets of size j, where none of them is the response
        result += (-1)**j * binomial(i-1, j) * log(2) * j
    subsets = binomial(n+1, j)
    here = (-1)**i * result * subsets
    sumSoFar += here #min(0,here)
    sumISoFar += i * here #min(0,here)
    synergySoFar += i * min(0, here)
    inp_subsets = binomial(n, j-1)
    fourier_mass_here = ((fourier_coefs[i]) ** 2) * inp_subsets
    fourier_mass_total += fourier_mass_here
    sensitivity += fourier_mass_here * (i-1)
    print("\t".join([str(round(x, 3)) for x in [i, (-1)**i * result, here, sumSoFar, sumISoFar, synergySoFar, fourier_coefs[i], fourier_mass_here, fourier_mass_total, sensitivity]]))
    # sumSoFar ends up at 0

print("TODO why is the last 'result' > 0, even though higher order MI is bounded by the individual entropies?")
