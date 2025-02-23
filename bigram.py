#%% Imports
import torch
from pprint import pprint
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc # nice colour map

#%% read in the data
words = open('names.txt').read().split()


# %% Bigram language model
"""
We model the probability of each letter given the previous letter, as a 2d tensor.
For the i,j entry, we have the probability of letter j given letter i.
Thus rows represent the previous letter, and columns represent the current letter.
"""
#%% Building a character encoder 
# get the sorted unique letters in words
chars = sorted(list(set("".join(words))))
# add special token for starting and ending each word
ST = "."
chars = [ST] + chars
# maps
ctoi = {s:i for i,s in enumerate(chars)} # maps chars to integers
itoc = dict(zip(ctoi.values(), ctoi.keys()))

pprint(ctoi)
# %% Counting bigrams
B = torch.zeros((len(ctoi), len(ctoi)), dtype=torch.int32)
for w in words:
    chs = ST + w + ST
    for c1, c2 in zip(chs, chs[1:]):
        i1 = ctoi[c1]
        i2 = ctoi[c2]
        B[i1, i2] +=1

#%% visualise this tensor
"""
In an earlier version, we used two special tokens for the start and end of each word.
We later combined them to just one.
"""
fig, ax = plt.subplots(figsize=(16,16))
ax.imshow(B)
color_thresh = int(B.max()/2)
# add the bigrams to each cell
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        chstr = itoc[i] + itoc[j]
        count = B[i,j].item()
        color="white"
        if count>color_thresh:
            color="black"
        # text uses "data coordinates", where "x" varies horisontally, and "y" vertically
        ax.text(j, i, int(B[i,j]), ha="center", va="top", color=color)
        ax.text(j, i, chstr, ha="center", va="bottom", color=color)

ax.axis("off")
plt.show()

#%% Sampling from the distribution
# Probability of each first letter given the special token start token
p_start = B[0].float()
p_start = p_start / p_start.sum() # normalise
p_start

#%% Making the normalised probabilties
"""
Tensor manipulations!
Summation dimensions, and broadcasting rules.

If we want to normalise each "row" of the B matrix, which is indexed by the first index,
this involves summing together all the elements in each row, i.e. a summation over the "column",
or second index. 

Summing over a dimension reduces that dimension to 1. This is apparent if you set keepdim=True.
If you have X.shape = 2,3, and X.sum(dim=1, keepdim=True), then its shape is (2,1).

Broadcasting is, for an operation on tensors, the tensors are automatically expanded
to be of equal sizes without making copies of the data.

Broadcasting semantics: https://pytorch.org/docs/stable/notes/broadcasting.html
Two tensors are "broadcastable" if the following rules hold:
- Each tensor has at least one dimension,
- When iterating over the dimension sizes, starting at the trailing dimension, 
    - The dimension sizes must be equal,
    - One of them is 1, or
    - One of them does not exist.

When we do
P = P / P.sum(dim=1, keepdim=True)
The shapes are:
27, 27
27, 1
When we divide, the second dimension of P.sum(...) is repeated 27 times,
and we divide the elements of P by P.sum(...).
"""
P = B.float()
P /= P.sum(dim=1, keepdim=True) # inplace operation uses less memory, unlike P = P / P.sum(...), which creates a copy of P

plt.imshow(P, cmap=cmc.acton) # check it is normalised along the correct dimension
plt.show()

print(P[0].sum()) # check first row is normalised

#%% Setting the seed
"""
The multinomial distribution is a distribution over k discrete events, each with associated probabilites,
and typically models the number of times you would expect to see each events, if you repeatedly sample an event
n times. In the case of torch.multinomial, we get integers distributed according to the probabilites p_start, so n=1.
"""
g = torch.Generator().manual_seed(214783647) # set the seed
# torch.multinomial(p_start, num_samples=20, replacement=True, generator=g) # sample first letters 

# %% Generate new names!
for _ in range(10): 
    ix = 0 # Start at the special token
    new_name = ""
    while True:
        p = P[ix] # pick out row of next character probs
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # sample index of new letter
        new_name += itoc[ix]
        if ix == 0: # break if it reaches special character
            break
    print(new_name)

# %% Figure out how to evaluate the model
"""
Maximum likelihood estimation.
Goal: maximise likelihood of the data w.r.t. the model parameters (statistical modeling)

Likelihood: The probability of the data-set under the model we have trained.
- Assuming independent data-points, product of the model predicted probability of each data-point
Log-likelihood.
- Allows the product to become the sum of the log probabilities.
- Large likelihood is big if data is very likely under the model.
Negative log-likelihood
- Frames this as a loss to minimise, as opposed to an objective to maximise.

Tips:
- Can look at the likelihood of individual names = product of the probabilites of bigrams.
- Some bigrams have probability zero, which has an infinite log-likelihood. To avoid this, smooth model by adding count of 1 bigram matrix.
"""
log_likelihood = 0.0
n = 0
for w in words:
    chs = ST + w + ST
    for c1, c2 in zip(chs, chs[1:]):
        n+=1
        i1 = ctoi[c1]
        i2 = ctoi[c2]
        prob = P[i1, i2] 
        logprob = torch.log(prob)
        log_likelihood+=logprob
        # print(f"{c1}{c2}: {prob:.4f} {logprob:.4f}")

nll = -log_likelihood
mean_nll = nll / n

print(f"Mean -log-likelihood: {mean_nll}, ideal 0")

# %%

