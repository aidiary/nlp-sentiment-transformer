import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        # 学習対象ではなく、既存の単語ベクトルを使う場合の書き方
        # freeze=Trueにするとパラメータ更新がされなくなる
        self.embeddings = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors,
                                                       freeze=True)

    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec


class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000**((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000**((2 * (i + 1)) / d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret


class Attention(nn.Module):
    def __init__(self, d_model=300):
        super(Attention, self).__init__()

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # self-attentionの場合は、q=k=v=x
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        mask = mask.unsqueeze(1)

        # maskはpadのところがFalse (0) になっている
        # 負の大きな値のマスクにするのはsoftmax(-inf)=0のため
        weights = weights.masked_fill(mask == 0, -1e9)

        normalized_weights = F.softmax(weights, dim=-1)

        output = torch.matmul(normalized_weights, v)
        output = self.out(output)

        return output, normalized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.attn = Attention(d_model)

        self.ff = FeedForward(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_normalized = self.norm_1(x)
        output, normalized_weights = self.attn(x_normalized, x_normalized, x_normalized, mask)

        x2 = x + self.dropout_1(output)

        x_normalized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normalized2))

        return output, normalized_weights


class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, output_dim=2):
        super(ClassificationHead, self).__init__()

        self.linear = nn.Linear(d_model, output_dim)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        # 各ミニバッチの各文の先頭の単語の特徴量を使って分類する
        # このように決めてlossを求めて訓練するのでそういう特徴が集まるように訓練される
        x0 = x[:, 0, :]
        out = self.linear(x0)
        return out


class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2):
        super(TransformerClassification, self).__init__()

        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model, max_seq_len)
        self.net3_1 = TransformerBlock(d_model)
        self.net3_2 = TransformerBlock(d_model)
        self.net4 = ClassificationHead(d_model, output_dim)

    def forward(self, x, mask):
        x = self.net1(x)
        x = self.net2(x)
        x, normalized_weights1 = self.net3_1(x, mask)
        x, normalized_weights2 = self.net3_2(x, mask)
        x = self.net4(x)
        return x, normalized_weights1, normalized_weights2


if __name__ == "__main__":
    from data import create_dataloader
    train_dl, val_dl, test_dl, TEXT = create_dataloader()
    batch = next(iter(train_dl))

    print('*** Embedder Test')
    net1 = Embedder(TEXT.vocab.vectors)
    x = batch.Text[0]
    print(x.shape)

    input_pad = 1
    # pad出ない部分がTrueになるmask
    input_mask = (x != input_pad)
    print(input_mask[0])

    x = net1(x)
    print(x.shape)

    print('*** PositionalEncoder Test')
    net2 = PositionalEncoder(d_model=300, max_seq_len=256)
    x = net2(x)
    print(x.shape)

    print('*** TransformerBlock Test')
    net3 = TransformerBlock(d_model=300)

    x, normalized_weights = net3(x, input_mask)
    print(x.shape)

    print('*** Transformer Test')
    net = TransformerClassification(TEXT.vocab.vectors, 300, 256, 2)
    x = batch.Text[0]
    out, normalized_weights1, normalized_weights2 = net(x, input_mask)
    print(out.shape)
    print(F.softmax(out, dim=1))
