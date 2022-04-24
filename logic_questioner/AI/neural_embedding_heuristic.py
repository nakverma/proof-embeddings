import torch.nn as nn
from torch import optim, Generator
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from sim_net import SimNet

import os
from time import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuralEmbeddingHeuristic:

    def __init__(self, model_file, is_state_dict=True):
        self.vocab_dict = {chr(i + 97): i + 1 for i in range(26)}
        self.vocab_dict.update({k: i + 27 for i, k in enumerate(
            ['V', '^', 'T', 'F', '(', ')', '~', '-', '<', '>', '=']
        )})
        self.vocab_dict["<PAD>"] = 0
        self.reverse_vocab = {v: k for k, v in self.vocab_dict.items()}

        if is_state_dict:
            self.model = SimNet(vocab_size=len(self.vocab_dict))
            self.model.load_state_dict(torch.load(model_file, map_location=device))
        else:
            self.model = torch.load(model_file, map_location=device)
        self.model.eval()

    def expr_to_vocab(self, expr):
        return torch.tensor([self.vocab_dict[i] for i in list(expr)])

    def sim_to_dist(self, sim):
        sim = torch.squeeze(sim)
        #print(sim, torch.argmax(sim))
        return torch.argmax(sim).item() # 1 / (sim + 1e-8)

    def embedding_dist(self, n1, n2):
        e1, e2 = self.expr_to_vocab(n1[0]), self.expr_to_vocab(n2[0])
        e1, e2, l1, l2 = e1[None, :], e2[None, :], torch.tensor([1]), torch.tensor([1])
        with torch.no_grad():
            sim, _, _ = self.model.forward((e1, e2, l1, l2), None, None)
        #print(f"Sim: {sim.item()}")
        return self.sim_to_dist(sim)#sim.item())


if __name__ == "__main__":
    n1, n2 = ('p', "Start"), ('p', None)
    n3, n4 = ('p->q', "Start"), ('p->qvr', None)
    n5, n6 = ('p->q', "Start"), ('~pvq', None)
    n7, n8 = ('p->q', "Start"), ('~q^~q^r^q', None)

    neh = NeuralEmbeddingHeuristic(os.path.join("models", "dist_model_parameters.pt"))
    print(neh.embedding_dist(n1, n2))
    print(neh.embedding_dist(n3, n4))
    print(neh.embedding_dist(n5, n6))
    print(neh.embedding_dist(n7, n8))
