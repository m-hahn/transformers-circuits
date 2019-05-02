# from Alexander Rush's annotated transformer


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
#import matplotlib.pyplot as plt
#import seaborn
#seaborn.set_context(context="talk")
#matplotlib inline

import random
import argparse


# This will work for small sequence lengths (e.g., 5).


parser=argparse.ArgumentParser()
parser.add_argument("--V", dest="V", type=int, default=3)
parser.add_argument("--beta1", dest="beta1", type=float, default=0.9)
parser.add_argument("--beta2", dest="beta2", type=float, default=0.98)
parser.add_argument("--eps", dest="eps", type=float, default=1e-9)
parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
#parser.add_argument("--warmup", dest="warmup", type=int, default=1000)
parser.add_argument("--batchSize", dest="batchSize", type=int, default=30)
parser.add_argument("--epochCount", dest="epochCount", type=int, default=20)
#parser.add_argument("--n_layers", dest="n_layers", type=int, default=3)
parser.add_argument("--lstm_dim", dest="lstm_dim", type=int, default=64)
#parser.add_argument("--d_ff_global", dest="d_ff_global", type=int, default=128)
#parser.add_argument("--h_global", dest="h_global", type=int, default=8)
#parser.add_argument("--dropout_global", dest="dropout_global", type=float, default=0.05)
parser.add_argument("--sequence_length", dest="sequence_length", type=int, default=10)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))

args=parser.parse_args()
print(args)

V = args.V
beta1 = args.beta1
beta2 =  args.beta2 #0.98
eps= args.eps #1e-9
lr = args.lr #1
#warmup= args.warmup #400
batchSize = args.batchSize #30
epochCount = args.epochCount #20
#n_layers = args.n_layers #1
lstm_dim = args.lstm_dim #2048
#h_global = args.h_global #8
#dropout_global = args.dropout_global #0.1
sequence_length = args.sequence_length


# https://discuss.pytorch.org/t/global-gpu-flag/17195
use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor 
                                             if torch.cuda.is_available() and x 
                                             else torch.FloatTensor)

use_gpu()


encoder = torch.nn.LSTM(lstm_dim, lstm_dim)
decoder = torch.nn.LSTM(lstm_dim, lstm_dim)

embeddings_enc = torch.nn.Embedding(V, lstm_dim)
embeddings_dec = torch.nn.Embedding(V, lstm_dim)

projection = torch.nn.Linear(lstm_dim, V)

components = [encoder, decoder, embeddings_enc, embeddings_dec, projection]

def parameters_():
   for c in components:
     for p in c.parameters():
        yield p
parameters = list(parameters_())

criterion = torch.nn.NLLLoss(reduction="none")


model_opt = torch.optim.Adam(parameters, lr=lr, betas=(beta1, beta2), eps=eps)

def forward(source, target_x, target_y):
   _, hidden = encoder(embeddings_enc(source.transpose(0,1)))
   output, _ = decoder(embeddings_dec(target_x.transpose(0,1)), hidden)
   logprobs = F.log_softmax(projection(output), dim=-1)
   x = logprobs
   y = target_y.transpose(0,1)
   loss = criterion(x.contiguous().view(-1, x.size(-1)), 
                    y.contiguous().view(-1))
   #print(loss)
   loss.mean().backward()

   _, predictions = torch.max(x.contiguous().view(-1, x.size(-1)), dim=1)

   global accuracies
   accuracy = (predictions == y.contiguous().view(-1)).float().mean()
   accuracies.append(float(accuracy))
   model_opt.step()
   model_opt.zero_grad()

   return loss.sum()


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = None
            self.ntokens = (self.trg_y != pad).data.sum()
    


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
  #      print("run_epoch", i)
        loss = forward(batch.src, batch.trg, batch.trg_y)
        total_loss += loss
        total_tokens += batch.ntokens.float()
        tokens += batch.ntokens.float()
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens.float(), tokens / elapsed))
            start = time.time()
            tokens = 0
#    print("done run_epoch")
    return total_loss / total_tokens



global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)



import random

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, sequence_length))).cuda()
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = (src.sum(dim=1) % 2 == 1).long().unsqueeze(1).repeat(1,2)
#        print(src)
 #       print(tgt)
        tgt[:,0] = 1
        yield Batch(src, tgt, 0)



class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = torch.nn.NLLLoss(reduce="sum") #criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()

        _, predictions = torch.max(x.contiguous().view(-1, x.size(-1)), dim=1)

        global accuracies
        accuracy = (predictions == y.contiguous().view(-1)).float().mean()
        accuracies.append(float(accuracy))
#        if random.random() > 0.95: # compute accuracy      
 #          print("ACCURACY", sum(accuracies)/len(accuracies))


        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
#        print(loss.data, loss.data.item(), norm)
        return loss.data.item() * norm.item()



# Train the simple copy task.

all_accuracies = []
for epoch in range(epochCount):
    print(epoch)
    accuracies = []
    run_epoch(data_gen(V, batchSize, 50), None, 
              SimpleLossCompute(None, None, model_opt))
#    print(run_epoch(data_gen(V, 30, 5), model, 
 #                   SimpleLossCompute(model.generator, criterion, None)))

    all_accuracies.append(sum(accuracies)/float(len(accuracies)))
    print("ACCURACY", all_accuracies[-1])



smoothed_best = sorted(all_accuracies)[-2:]
smoothed_best = sum(smoothed_best) / len(smoothed_best)
with open("logs/per_run/"+__file__+"_model_"+str(args.myID)+".txt", "w") as outFile:
   print(sum(accuracies)/float(len(accuracies)), file=outFile)
   print(smoothed_best, file=outFile)

print(sum(accuracies)/float(len(accuracies)))
print(smoothed_best)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys



