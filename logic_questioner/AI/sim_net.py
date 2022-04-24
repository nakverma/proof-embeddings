import torch.nn as nn
from torch import optim, Generator
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, random_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_dist = 11


class SimNet(nn.Module):

  def __init__(self, vocab_size, embed_dim=30, hidden_dim=100,
               output_dim=max_dist, layer_dim=1):
    super(SimNet, self).__init__()
    self.layer_dim = layer_dim
    self.hidden_dim = hidden_dim

    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim,
                      num_layers=layer_dim, batch_first=True)
    self.ffn = nn.Sequential(
        nn.Linear(hidden_dim*2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Softmax(dim=1)
    )

  def forward_one(self, x, x_lens, h):
    #print(x.shape)
    x = self.embedding(x)
    #print(x.shape, torch.sum(x, dim=2).shape)
    xp = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False).to(device)
    # print(xp.batch_sizes)
    out_packed, h = self.gru(xp, h)
    out, out_len = pad_packed_sequence(out_packed, batch_first=True)
    return out, h

  def forward(self, X, h1, h2):
    w1, w2, l1, l2 = X
    #print([reverse_vocab[i.item()] for i in w1[0]], [reverse_vocab[i.item()] for i in w2[0]])
    #print(w1.shape, w2.shape)
    if h1 is None:
      h1 = torch.randn(self.layer_dim, w1.size(1), self.hidden_dim).requires_grad_().to(device)
    if h2 is None:
      h2 = torch.randn(self.layer_dim, w1.size(1), self.hidden_dim).requires_grad_().to(device)

    w1_out, h1 = self.forward_one(w1, l1, h1)
    w2_out, h2 = self.forward_one(w2, l2, h2)
    #print(w1_out.shape, w2_out.shape)
    w1_out, w2_out = w1_out.sum(dim=1), w2_out.sum(dim=1) # w1_out[:,-1,:], w2_out[:,-1,:] # choose last hidden state
    #print(w1_out.shape, w2_out.shape)

    join = torch.cat((w1_out,w2_out), dim=1)
    #print(join.shape)
    pred = self.ffn(join)
    #print(pred.shape)
    return pred, h1, h2

"""
class SimNet(nn.Module):

    def __init__(self, vocab_size, embed_dim=10, hidden_dim=50,
               output_dim=50, layer_dim=1):
        super(SimNet, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim,
                          num_layers=layer_dim, batch_first=True)
        self.proj1 = nn.Linear(hidden_dim, output_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x, x_lens, h):
        #print(x.shape)
        x = self.embedding(x)
        #print(x.shape, torch.sum(x, dim=2).shape)
        xp = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False).to(device)
        # print(xp.batch_sizes)
        out_packed, h = self.gru(xp, h)
        out, out_len = pad_packed_sequence(out_packed, batch_first=True)
        return out, h

    def forward(self, X, h1, h2):
        w1, w2, l1, l2 = X
        #print([reverse_vocab[i.item()] for i in w1[0]], [reverse_vocab[i.item()] for i in w2[0]])
        #print(w1.shape, w2.shape)
        if h1 is None:
          h1 = torch.randn(self.layer_dim, w1.size(1), self.hidden_dim).requires_grad_().to(device)
        if h2 is None:
          h2 = torch.randn(self.layer_dim, w1.size(1), self.hidden_dim).requires_grad_().to(device)

        w1_out, h1 = self.forward_one(w1, l1, h1)
        w2_out, h2 = self.forward_one(w2, l2, h2)
        #print(w1_out.shape, w2_out.shape)
        w1_out, w2_out = w1_out.sum(dim=1), w2_out.sum(dim=1) # w1_out[:,-1,:], w2_out[:,-1,:] # choose last hidden state
        #print(w1_out.shape, w2_out.shape)
        p1, p2 = self.proj1(w1_out), self.proj2(w2_out)
        # p1, p2 = p1 / torch.norm(p1), p2 / torch.norm(p2)  # normalize?
        #print(p1[:,None,:].shape, p2[:,:,None].shape)
        sim = torch.bmm(p1[:,None,:], p2[:,:,None])  # dot product
        #print(sim.shape, sim)
        sim = self.sigmoid(sim)
        #print(sim.shape, sim)

        return sim, h1, h2



"""