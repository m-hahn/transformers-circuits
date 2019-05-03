# 10: Important: This is done conditioned on sequence lenth (which is restricted when sampling new points)

# Good results: logs/search-10-transformer.py_model_584553391.txt
# Search command ./python36 search-10.py 1 10000 logs/search-10-transformer.py_model_986314965.txt 0.0002
import subprocess
import random

from math import exp


import random

myID = random.randint(0,1000000000)


import sys

gpus = 1
numberOfJobs = int(sys.argv[1])
limit = int(sys.argv[2]) #if len(sys.argv) > 7 else 100

priorKnowledge = sys.argv[3] if len(sys.argv)>3 else None
if priorKnowledge == "NONE":
    priorKnowledge = None
noiseVariance = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02

#MODEL_TYPE = sys.argv[5] if len(sys.argv)>5 else  "RANDOM_BY_TYPE"
#assert MODEL_TYPE == "RANDOM_BY_TYPE", "are you sure?"


#if priorKnowledge is not None:
#   assert MODEL_TYPE in priorKnowledge



import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return expected_improvement

n_iters = 10
sample_loss = None


bounds = []
#bounds.append(["dropout1", float] + [x/20.0 for x in range(10)])
#bounds.append(["emb_dim", int, 50, 100, 150, 200, 300])
#bounds.append(["lstm_dim", int, 64, 128, 256, 512, 1024])
#bounds.append(["layers", int, 1, 2, 3])
#bounds.append(["learning_rate", float, 0.001, 0.005, 0.01, 0.05, 0.1])
#bounds.append(["dropout2", float] + [x/20.0 for x in range(10)])
#bounds.append(["batch_size", int, 2, 4, 8, 16, 32, 64])
#bounds.append(["replaceWordsProbability", float] + [x/20.0 for x in range(10)]) 

import math
#bounds.append(["ENTROPY_WEIGHT", float] + [0.001, 0.005, 0.01, 0.05, 0.1])
#bounds.append(["DROPOUT", float] + [x/20.0 for x in range(10)]) # = float(sys.argv[2])
#bounds.append(["ENTROPY_LISTENER", float, 0.0, 0.001, 0.005, 0.01, 0.05, 0.1])
#bounds.append(["LEARNING_RATE_SPEAKER", float, 0.0001, 0.0002, 0.0005, 0.001,0.002, 0.003, 0.003,0.003, 0.005])
#bounds.append(["N_HIDDEN", int, 4, 8, 16, 32, 64, 128, 256, 512]) # = 256
#bounds.append(["N_BATCH", int, 64, 128, 256, 512]) #= 256
#bounds.append(["TRAINING_ITERATIONS", int, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000])
#bounds.append(["NOISING", float] + [0]) # = float(sys.argv[2])
#bounds.append(["SHUFFLE", float] + [x/20.0 for x in range(10)]) # = float(sys.argv[2])
#bounds.append(["N_VOCAB", int] + [5]) #4,7,10,14,20]) # = float(sys.argv[2])
#bounds.append(["LEARNING_RATE_LISTENER", float, 0.0001, 0.0002, 0.0005, 0.001,0.002, 0.003, 0.003,0.003, 0.005])
#bounds.append(["COLORS_PARTS_NOISE_RATE", float] + [x/20.0 for x in range(10)]) # = float(sys.argv[2])
#bounds.append(["QUANTITY_NOISE_RATE", float] + [x/20.0 for x in range(10)]) # = float(sys.argv[2])




bounds.append(["V", int, 4])
bounds.append(["beta1", float, 0.9, 0.95])
bounds.append(["beta2", float, 0.95, 0.98])
bounds.append(["eps", float, 1e-9])
bounds.append(["factor", float, 1.0])
bounds.append(["warmup", int, 10, 50, 100, 200, 400, 500, 1000, 1000, 2000])
bounds.append(["batchSize", int, 30,40,50,60, 200, 400, 500, 800, 1000])
bounds.append(["epochCount", int, 5, 10, 15, 20,30, 50, 100, 200, 300, 400])
bounds.append(["n_layers", int, 1])
bounds.append(["d_model_global", int, 16, 32, 64])
bounds.append(["d_ff_global", int, 16, 32, 64])
bounds.append(["h_global", int, 1, 2,4])
bounds.append(["dropout_global", float, 0.0, 0.05, 0.1])
bounds.append(["sequence_length", float, 40, 60, 80, 100, 200]) #, 15, 20, 30, 100, 1000])






argumentNames = [x[0] for x in bounds]




#x0=[0.5] * len(names)

values = [x[2:] for x in bounds]
names = [x[0] for x in bounds]

import random

def sample():
   while True:
     result = [random.choice(values[i]) for i in range(len(bounds))]
#     if result[names.index("lstm_dim")] == 1024 and result[names.index("layers")] == 3:
#        continue
     if result[names.index("sequence_length")] not in [200]:
        continue
     if result[names.index("batchSize")] >= 500:
        continue
#     if result[names.index("dropout_global")] == 0.0:
#        continue
     return result

def represent(x):
   result = [float(values[i].index(x[i]))/len(values[i]) for i in range(len(x))]
   return result
  

n_pre_samples=5
gp_params=None
random_search=False
alpha=noiseVariance # 0.0025
epsilon=1e-7

xp_raw = []
y_list = []

def format(x):
   try:
     return int(x)
   except ValueError:
     return float(x)

if priorKnowledge is not None:
  with open(priorKnowledge, "r") as inFile:
    for line in inFile:
      line = line.strip().split("\t")
      line[1] = list(map(float, line[1][1:-1].split(",")))
      for y in line[1]:
         y_list.append(y)
         xp_raw.append(list(map(format,line[2:])))
print(xp_raw)

# 4.699497452695695	[4.66408287849234]	0.35	200	128	1	0.005	0.3	18



kernel = gp.kernels.Matern()
model = gp.GaussianProcessRegressor(kernel=kernel,
                                    alpha=alpha,
                                    n_restarts_optimizer=10,
                                    normalize_y=True)

theirGPUs = []
perGPU = ([0]*gpus)
runningProcesses = []
theirIDs = []
theirXPs = []
positionsInXPs = []

version = "15-transformer.py"

myOutPath="logs/search-"+version+"_model_"+str(myID)+"_batch.txt"
IDsForXPs = []


def extractArguments(x):
   result = []
   for i in range(len(argumentNames)):
      result.append("--"+argumentNames[i])
      result.append(x[names.index(argumentNames[i])])
   return result

import os
import subprocess


def getResult(i):
#   return theirXPs[i][0]
   if runningProcesses[i].poll() is not None:
      with open("logs/per_run/"+version+"_model_"+str(theirIDs[i])+".txt", "r") as inFile:
         loss = -float(next(inFile).strip())
         return loss
   else:
      return None 

import time

posteriorMeans = []


if len(xp_raw) > 0:
   print("FITTING MODEL")
   model.fit(list(map(represent, xp_raw)), y_list)
   print("DONE")
yp_filtered = y_list + [100]

#for n in range(n_iters):
while True:
    assert len(runningProcesses) == len(theirIDs)
    assert len(runningProcesses) == len(positionsInXPs)
    assert len(runningProcesses) == len(theirXPs)
    assert len(runningProcesses) == len(theirGPUs)
#    print "PROCESSES"
#    print runningProcesses
#    print theirIDs

    canReplace = None
    if len(runningProcesses) >= numberOfJobs: # wait until some process terminates
       for i in range(len(runningProcesses)):
          loss = getResult(i)
          if loss is not None:
              canReplace = i
              y_list[positionsInXPs[i]] = loss
              break
       if canReplace is None:
         print("Sleeping")
         print(myOutPath)
         time.sleep(5)
         print("Checking again")
         continue
       del runningProcesses[canReplace]
       del theirIDs[canReplace]
       del positionsInXPs[canReplace]
       del theirXPs[canReplace]
       perGPU[theirGPUs[canReplace]] -= 1
       assert perGPU[theirGPUs[canReplace]] >= 0
       del theirGPUs[canReplace]
       print("OBTAINED RESULT")

    if len(posteriorMeans) > 50 and random.random() > 0.99:
       print("Sampling old point, to see whether it really looks good")
#       print posteriorMeans
       nextPoint = random.choice(posteriorMeans[:100])[2]
 #      print nextPoint
  #     quit()
    else:        
#       if len(runningProcesses) < numberOfJobs:
       if len(xp_raw) - numberOfJobs < 100: # choose randomly until we have 20 datapoints to base our posterior on
          print("Choose randomly")
          nextPoint = sample()
       else:
          samples = [sample() for _ in range(1000)]
          print("Starting computing GP")
          acquisition = expected_improvement(np.array([ represent(x) for x in samples]), model, yp_filtered, False, len(bounds)) 
          print("Ending computing GP")
          best = np.argmax(acquisition)
          nextPoint = samples[best]

    print("NEW POINT")
    print(nextPoint)

    mu, sigma = model.predict(np.array(represent(nextPoint)).reshape(-1, len(bounds)), return_std=True)
    print(mu)
    
    # create an ID for this process, start it
    idForProcess = random.randint(0,1000000000)


    
    my_env = os.environ.copy()

#    quit()
#    subprocess.call(command)
    FNULL = open(os.devnull, "w")
#    p = None
    gpu = np.argmin(perGPU)
    print("GPU "+str(gpu)+" out of "+str(gpus))
    perGPU[gpu] += 1

    command = list(map(str,["./python36", version] + extractArguments(nextPoint) + ["--myID", idForProcess]))
    print(" ".join(command))
    #quit()
    #my_env["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu])
    p = subprocess.Popen(command, stdout=FNULL, env=my_env) # stderr=FNULL, 
    runningProcesses.append(p)
    theirIDs.append(idForProcess)
    theirXPs.append(nextPoint)
    IDsForXPs.append(idForProcess)
    theirGPUs.append(gpu)
    print("ALLOCATED GPUs")
    print(theirGPUs)
#    sampledResult = 
#    x_to_predict = x.reshape(-1, n_params)
#
    mu, sigma = model.predict(np.array(represent(nextPoint)).reshape(-1, len(bounds)), return_std=True)
    sampledResult = np.random.normal(loc=mu, scale=sigma)


    # Update lists
    positionsInXPs.append(len(xp_raw))
    xp_raw.append(nextPoint)
    y_list.append(sampledResult)

    
    xp_raw_filtered = []
    y_list_filtered = []

    for i in range(len(xp_raw)):
        if i in positionsInXPs:
           continue
        xp_raw_filtered.append(xp_raw[i])
        y_list_filtered.append(y_list[i])
    
    xp_filtered = np.array(list(map(represent, xp_raw_filtered))).reshape(len(xp_raw_filtered), len(bounds))
    yp_filtered = np.array(y_list_filtered)



    print("USING")
#    print(xp_raw_filtered)
 #   print(xp_filtered)
  #  print(IDsForXPs)
   # print(yp_filtered)
    if len(xp_raw_filtered) > 0:
       print("FITTING THE MODEL")
       model.fit(xp_filtered, yp_filtered)
       print("DONE")
     
       # find setting with best posteriori mean
       posteriorMeans = {}
       posteriorMu_batch, posteriorSigma_batch = model.predict(np.array([represent(xp_raw[i]) for i in range(len(xp_raw))] ).reshape(len(xp_raw), len(bounds)), return_std=True)

       
       for i in range(len(xp_raw)):
           if i in positionsInXPs:
              continue
           if str(xp_raw[i]) not in posteriorMeans:
             posteriorMu, posteriorSigma = posteriorMu_batch[i], posteriorSigma_batch[i] #model.predict(np.array(represent(xp_raw[i])).reshape(-1, len(bounds)), return_std=True)
             # sort by upper 95 \% confidence bound
             posteriorMeans[str(xp_raw[i])] = (posteriorMu, [y_list[i]], xp_raw[i], posteriorMu-2*posteriorSigma, posteriorMu+2*posteriorSigma)
           else:
             posteriorMeans[str(xp_raw[i])][1].append(y_list[i])
       posteriorMeans = [posteriorMeans[x] for x in posteriorMeans]
       posteriorMeans = sorted(posteriorMeans, key=lambda x:x[4]) # sort by upper confidence bound
       print("Best Parameter Settings")
       print(posteriorMeans)
       print(myOutPath)
       with open(myOutPath, "w") as outFile:
          print("\n".join(list(map(lambda x:"\t".join(map(str,[x[0], x[1]] + x[2])), posteriorMeans))), file=outFile)
#       posteriorMeans = sorted(posteriorMeans, key=lambda x:x[3]) # sort by lower confidence bound
       posteriorMeans = sorted(posteriorMeans, key=lambda x:x[0]) # sort by expectation
 
    if len(posteriorMeans) > limit:
        print(myOutPath)
        break

    xp = np.array(list(map(represent, xp_raw))).reshape(len(xp_raw), len(bounds))
    yp = np.array(y_list)

    print("FITTING THE MODEL")
    model.fit(xp, yp)
    print("DONE")


quit()











