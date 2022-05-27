import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class AttentionHead(nn.Module):
    """ single attention head """
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.linear = nn.Linear(d, d)
        self.layer_norm = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.position_l1 = nn.Linear(d, d)
        self.position_l2 = nn.Linear(d, d)

    def forward(self, x):
        # b: batch, k: sequence length, d: embedding dim
        b, k, d = x.shape

        # shape(q) = b * k * d
        query = self.q(x)

        # shape(k) = b * k * d
        key = self.k(x)

        # shape(v) = b * k * d
        value = self.v(x)

        # shape(score) = b * k * k
        score = query @ key.transpose(-1, -2)

        # shape(attention) = b * k * k
        attention = torch.softmax(score / np.sqrt(d * 1.), dim=-1)

        # shape(lookup) = b * k * d
        lookup = attention @ value

        # shape(appy_linear) = b * k * d
        apply_linear = self.linear(lookup)

        # shape(layer_normed) = b * k * d
        layer_normed = self.layer_norm(apply_linear + x)

        # position wise ff
        position_wise = self.position_l2(F.relu(self.position_l1(layer_normed)))

        # add and norm again
        added_p_wise = self.layer_norm2(position_wise + layer_normed)

        return added_p_wise, attention


def embed(word):
    all_words = []
    for word in word.split(' '):
        if word == "the":
            c = torch.tensor([1, 0, 0, 0])
        elif word == "dog":
            c = torch.tensor([0, 1, 0, 1])
        elif word == "cat":
            c = torch.tensor([0, 0, 1, 0])
        elif word == "runs":
            c = torch.tensor([1, 1, 0, 0])
        elif word == "sits":
            c = torch.tensor([1, 0, 1, 1])
        else:
            raise NameError(f"not found {word}")

        all_words.append(c)
    return torch.stack(all_words)


def get_labels(sentences):
    """ program rules """
    labels = torch.tensor([[0, 1] if "dog" in s and "the" in s else [1, 0] for s in sentences]).float()
    return labels


class AttentionPlusFF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = AttentionHead(d)
        self.linear = nn.Linear(12, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, attention = self.attention(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.softmax(x)
        return x, attention


def main():

    # Some hardcoded sentence embeddings
    sentences = ["the dog sits", "dog sits the", "cat dog sits",
                 "the dog runs", "dog cat runs", "the cat runs",
                 "runs cat the", "cat cat cat", "dog dog dog",
                 "the the the", "dog the cat", "dog cat the",
                 "dog dog dog"]
    embeddings = torch.stack([embed(s) for s in sentences]).float()
    labels = get_labels(sentences)

    # Init
    d = 4
    my_model = AttentionPlusFF(d=d)

    my_model.train()
    opt = torch.optim.AdamW(my_model.parameters(), lr=0.01)
    loss = nn.BCELoss()

    for i in range(500):
        opt.zero_grad()
        permutation = torch.randperm(embeddings.shape[0])

        embeddings = embeddings[permutation]
        labels = labels[permutation]

        forward, _ = my_model(embeddings)
        output = loss(forward, labels)

        print(f"iter {i} | {output}")

        output.backward()
        opt.step()


if __name__ == '__main__':
    main()
