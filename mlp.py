"""
Three character language model that predicts the next character.
Based on the paper, "A neural probabilistic language model", Bengio et al. 2003. 
"""
#%% imports
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#%% data-set
words = open('names.txt').read().split() # read in the names
# Building a character encoder 
chars = sorted(list(set("".join(words))))
ST = "."
chars = [ST] + chars
# maps
ctoi = {s:i for i,s in enumerate(chars)} # maps chars to integers
itoc = dict(zip(ctoi.values(), ctoi.keys()))

block_size = 3 # context length: how many characters do we take to predict the next one

X, Y = [], []
for w in words:
    context = [ctoi[ST]] * block_size # make padding
    for ch in w + ST:
        ix = ctoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix] # move block one charcter along

X = torch.tensor(X)
Y = torch.tensor(Y)

#%% Look at the first few examples
for i in range(5):
    x_ex = "".join([itoc[ch.item()] for ch in X[i]])
    y_ex = itoc[Y[i].item()]
    print(f"{x_ex}->{y_ex}")


#%% build the model
C = torch.randn((27,2))
W1 = torch.randn((6,100))
b1 = torch.randn(100)
W2 = torch.randn((100, 27))
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]
for param in parameters:
    param.requires_grad = True
"""
We want to concatenate the three character embeddings, 
In memory, tensors are a one-dimensional tensor. You see this by calling a.storage().
Blogpost: http://blog.ezyang.com/2019/05/pytorch-internals/
We can 
"""
#%% forward pass
emb = C[X] # N, 3, 2
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
#%% negative log likelihoods
logits = h @ W2 + b2
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
loss = -probs[torch.arange(len(X)), Y].log().mean()
#%% effecient torch implementation
"""
When we calculate the negative log-likelihood as above, it is ineffecient because we have to save each intermediate step into memory.
By calling F.cross_entropy, they are also able to succincly calculate the gradient of cross-entropy loss on the backward pass,
as opposed the gradients of each step how we calculate it above.
A fused kernel is, according to Claude, an optimisation technique where multiple operations are combined into a single GPU kernel.
For example, a linear layer followed by relu is often done sequentially. One could combine these operations
into a single step. This can make the forward and backward pass much more effecient.
Another issue with our implementation is overflow with logits.exp() for large logits.
F.cross_entropy deals with this by subtracting the largest value from all the logits, which does not alter
the probabilities, since exp(a + c) = exp(a)exp(c)
"""
loss = F.cross_entropy(logits, Y)

# %% Training in a loop
"""
Improvements we can make to the training code are
- finding a reasonable learning rate.
- using minibatches

Fiding a reasonable learning rate:
- Start by trying different values on a small number of steps and identify
    - a lower bound: loss decreases, but not by much
    - an upper bound: going larger means loss no longer decreases
"""

g = torch.Generator().manual_seed(214783467)
C = torch.randn((27,2), generator=g)
W1 = torch.randn((6,100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]
for param in parameters:
    param.requires_grad = True

epochs = 10000
batch_size = 32

# lr tuning
lr = 10**-1
lre = torch.linspace(-3, 0, epochs)
lrs = 10**lre
lre_i = []
loss_i = [] 

for i in range(epochs):
    ix = torch.randint(0, len(X), size=(batch_size,))
    # Forward pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item(), end="\r")

    # Backward pass
    for param in parameters:
        param.grad = None
    loss.backward()
    
    # lr = lrs[i]
    for param in parameters:
        param.data -= lr*param.grad # if we didn't do .data, autograd would track this as an operation 

    # # track stats
    # lre_i.append(lre[i])
    # loss_i.append(loss.item())

# %% Visualise learning rate
"""
Plot shows that the minimum of the loss was achieved at a learning rate of around 10^-1.
Tips for production:
- find the learning rate in the way above
- train for a long time with this learning rate
- then decay the learning rate at the end to obtain a trained model.
(how does this compare to other lr schedulers)
"""
plt.plot(lre_i, loss_i)
plt.show()

# %% 
"""
Even though we achieve a lower loss, i.e. negative log-likelihood, this can also reflect overfitting.
This is why we typically define train, val and test splits. 
"""
def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        context = [ctoi[ST]] * block_size
        for ch in w + ST:
            ix = ctoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

words = open('names.txt').read().split() # read in the names
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# %% Train, val, test
g = torch.Generator().manual_seed(214783467)
C = torch.randn((27,2), generator=g)
W1 = torch.randn((6,100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]
for param in parameters:
    param.requires_grad = True



#%% Train
steps = 10000
batch_size = 128
lr = 10**-2

for i in range(steps):
    ix = torch.randint(0, len(Xtr), size=(batch_size,))
    # Forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # Backward pass
    for param in parameters:
        param.grad = None
    loss.backward()
    
    # Update
    for param in parameters:
        param.data -= lr*param.grad # if we didn't do .data, autograd would track this as an operation 

#%% Evaluate
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss.item(), end="\r") 

#%% Scaling up model latent dimensions
"""
To further improve performance, Andrej considers increasing the latent dimension of character embeddings
and the size of the middle layer. At some point the training and validation losses diverged.
He introduces the idea of hyperparameter tuning.
"""
for _ in range(10):
    context = [ctoi[ST]]*3
    out = ""
    while True:
        # Sample from the updated model
        x_new = torch.tensor([context]) # 1, 3
        emb = C[x_new]
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        ch = itoc[ix] 
        out += ch
        if ch == ST:
            print(out)
            break
    # %%
