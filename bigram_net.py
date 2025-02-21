"""
Create a bigram model (predicts the probability of a particular letter given the previous letter)
by creating a training-set of pairs of adjacent letters and training a neural network.
"""
#%% Imports
import torch
import torch.nn.functional as F
from typing import List
from pprint import pprint
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc # nice colour map

#%% Read in the data
words = open('names.txt').read().split()

#%% Building a character encoder 
# get the sorted unique letters in words
chars = sorted(list(set("".join(words))))
# add special token for starting and ending each word
ST = "."
chars = [ST] + chars
# maps
ctoi = {s:i for i,s in enumerate(chars)} # maps chars to integers
itoc = dict(zip(ctoi.values(), ctoi.keys()))

#%% Create training set
xs, ys = [], []

for w in words: # just focus on first word
    chs = ST + w + ST
    for c1, c2 in zip(chs, chs[1:]):
        i1 = ctoi[c1]
        i2 = ctoi[c2]
        xs.append(i1)
        ys.append(i2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
inds = torch.arange(xs.nelement())

"""
Note:
- torch.tensor constructs a tensor and infers the dtype, torch.Tensor returns a float tensor.

One-hot encoding for the integer data-sets we have.
"""
xenc = F.one_hot(xs, num_classes=27).float()
yenc = F.one_hot(xs, num_classes=27).float()
#%% Simple linear transform of the inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g) # initialise weights
logits = xenc @ W
counts = logits.exp() # softmax
probs = counts / counts.sum(1, keepdim=True) 

#%% Go through the -log-likelihood of each bigrams
for i in range(len(xs[:5])):
    xi = xs[i]
    yi = ys[i]
    ps = probs[i] # probabilities predicted
    py = ps[yi] # likelihood of correct answer
    nll = -torch.log(py)
    print(f"{itoc[xi.item()]}{itoc[yi.item()]}, p(y|x)={py.item():.4f}")
    print(f"nll: {nll:.4f}")


#%% Init weights
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True) # initialise weights

#%% Do gradient descent
for epoch in range(10):
    # Forward pass
    logits = xenc @ W
    counts = logits.exp() # softmax
    probs = counts / counts.sum(1, keepdim=True) 
    loss = - probs[inds, ys].log().mean() + 0.01*(W**2).mean()
    print(loss)
    # Backward pass
    W.grad = None # set the gradients to zero
    loss.backward()
    # Update
    W.data += -50*W.grad


# %%
# Visualise gradients
plt.imshow(W.grad, cmap=cmc.acton)
plt.show()

#%% Visualise "counts"
plt.imshow(W.data.exp(), cmap=cmc.acton)
plt.show()

"""
We have used gradient-based optimisation to train a bigram model.
If you visualise the count matrix from the bigram.py script, and the exponentiated
weight matrix here, you actually end up with a similar picture.

Also, the mean -log(likelikood) over all bigrams in the data-set in the previous notebook
was 2.4541, and using gradient-descent, we get around 2.4648. 

We can add l2 regularisation, similar to the smoothing trick we did in the bigram model.
"""

#%% Sampling from the neural network
g = torch.Generator().manual_seed(214783647) # set the seed

# %% Generate new names!
for _ in range(10): 
    ix = 0 # Start at the special token
    new_name = ""
    while True:
        # Before:
        # p = P[ix] # pick out row of next character probs
        # ========
        # Now:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        # Forward pass
        logits = xenc @ W
        counts = logits.exp() # softmax
        probs = counts / counts.sum(1, keepdim=True) 
        # Sample
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        new_name += itoc[ix]
        if ix == 0: # break if it reaches special character
            break
    print(new_name)

#%%
ix = 0 # Start at the special token
#%%
xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
#%% Forward pass
logits = xenc @ W
counts = logits.exp() # softmax
probs = counts / counts.sum(1, keepdim=True) 
ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
itoc[ix]

# %%
