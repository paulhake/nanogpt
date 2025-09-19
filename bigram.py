import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup device for torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Hyperparameters
torch.manual_seed(1337)
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200


# Read input text
try:
    with open('input.txt','r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Error: 'input.txt' file not found. Please ensure the file exists in the current directory.")
    exit(1)
except IOError:
    print("Error: Unable to read 'input.txt' file. Please check file permissions.")
    exit(1)

print(f"length of text file is:{len(text)}")

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print("all the unique characters:", ''.join(chars))
#print("vocab size:", vocab_size)

# Create tokenizer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder:

#print(encode("hii there"))
#print(decode(encode("hii there")))

# Convert text to tensor
data = torch.tensor(encode(text), dtype=torch.long)
#print(f'data tensor shape is: {data.shape}, {data.dtype}')
#print(data[:1000])

# Train and val split
n = int(0.9*len(data)) # first 90% will be train
train_data = data[:n]
val_data = data[n:]
#print(train_data.shape, val_data.shape)

#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # pick random starting indices for the batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # for each index, get the block of data of size block_size
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # and the targets are just one character ahead of inputs
    x, y = x.to(device), y.to(device)
    return x, y

#xb, yb = get_batch('train')
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#simple bigram model
# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # each token directly reads off the logits for the next token from a lookup table

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C) batch, time, channel (vocab size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # get the predictions
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create optimizer and train
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))