"""
Three character language model that predicts the next character.
Based on the paper, "A neural probabilistic language model", Bengio et al. 2003. 
"""
#%% imports
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

X_enc = F.one_hot(X, num_classes=27).float() # one-hot, shape [228146, 3, 27]
Y_enc = F.one_hot(Y, num_classes=27) # shape [228146, 27]
inds = torch.arange(len(X_enc))
#%% Look at the first few examples
for i in range(5):
    x_ex = "".join([itoc[ch.item()] for ch in X[i]])
    y_ex = itoc[Y[i].item()]
    print(f"{x_ex}->{y_ex}")

# %% Make the model
"""
My attempt at making the model failed because I was not sure how I could manually apply the gradients.
Model:
- Look up table for each of the characters that maps them to an embedding.
- Take the three embeddings (concatenate them?), and then pass it through a fully connected layer with a tanh non-linearity
- Finally, apply a softmax function, outputting a distribution over the next letters.
"""
class NeuralLM(torch.nn.Module):
    def __init__(
        self,
        ch_embed_dim: int = 5,
        h_dim: int = 10
    ):
        super().__init__()
        self.ch_embed_dim = ch_embed_dim
        self.h_dim = h_dim
        g = torch.Generator().manual_seed(2147483647)
        
        # Create and assign parameters directly
        self.embedding = torch.nn.Parameter(
            torch.randn((27, self.ch_embed_dim), generator=g)
        )
        
        self.weights = torch.nn.Parameter(
            torch.randn((3*self.ch_embed_dim, self.h_dim), generator=g)
        )
        
        self.biases = torch.nn.Parameter(
            torch.randn((1, self.h_dim), generator=g)
        )
        
        self.s_weights = torch.nn.Parameter(
            torch.randn((self.h_dim, 27), generator=g)
        )
    
    def forward(self, x):
        """
        x: shape N, 3, 27
        """
        # N, 3, 27
        #       27 x ch_embed_dim
        # N, 3, ch_embed_dim
        embeds = x@self.embedding
        # concat the last channels: N, 3x
        N, _, _ = embeds.shape
        embeds = embeds.reshape(N,-1)
        # through weights
        w = embeds @ self.weights 
        wb = w + self.biases
        # nonlineartiy
        wb = F.tanh(wb)

        # softmax
        logits = wb@self.s_weights
        logits = logits.exp()
        logits = logits / logits.sum(dim=1, keepdim=True) # (N, 1)

        return logits

#%%
nlm = NeuralLM()
# %%
"""
Note, yesterday, lowest loss was -log likelihood of 2.4648.
Loss stops decreasing for lr > 1.
"""
for _ in range(500):
    # Forward pass
    probs = probs = nlm(X_enc)
    loss = - probs[inds, Y].log().mean() 
    print(loss)
    # Backward pass
    nlm.zero_grad()
    loss.backward()
    # Update parameters
    gradients = {name: param.grad.clone() for name, param in nlm.named_parameters() if param.grad is not None}
    with torch.no_grad():
        for name, param in nlm.named_parameters():
            if name in gradients:
                param -= 1 * gradients[name]

#%% visualise the paramters
for name, grads in gradients.items():
    plt.imshow(grads)
    plt.title(name)
    plt.show()

# %% sample names
chunk = [[ix]*3 for _ in range(10)]
new_names = ["" for _ in range(10)]

#%%
xenc = F.one_hot(torch.tensor(chunk), num_classes=27).float()
# Forward pass
probs = nlm(xenc) # 10, 27
# Sample
ixs = torch.multinomial(probs, num_samples=1, replacement=True) # samples from distribution of each row, so returns 10x1
new_names = [name + itoc[ix.item()] for ix, name in zip(ixs, new_names)]
chunk = [ chu[1:] + [ix.item()] for ix, chu in zip(ixs, chunk)]

print(new_names)
# %%
