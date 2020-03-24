import string
import re
import random
import torchtext
from torchtext.vocab import Vectors


def preprocessing_text(text):
    text = re.sub('<br />', '', text)
    for p in string.punctuation:
        if p == '.' or p == ',':
            continue
        else:
            text = text.replace(p, ' ')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    return text


def tokenizer_punctuation(text):
    return text.strip().split()


def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


def create_dataloader(max_length=256, batch_size=24):
    TEXT = torchtext.data.Field(sequential=True,
                                tokenize=tokenizer_with_preprocessing,
                                use_vocab=True,
                                lower=True,
                                include_lengths=True,
                                batch_first=True,
                                fix_length=max_length,
                                init_token='<cls>',
                                eos_token='<eos>')
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(path='./data/',
                                                                 train='IMDb_train.tsv',
                                                                 test='IMDb_test.tsv',
                                                                 format='tsv',
                                                                 fields=[('Text', TEXT),
                                                                         ('Label', LABEL)])
    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))

    # ボキャブラリのセット
    # この時点では単語IDを返すだけでベクトル表現は使われない（メモリ効率のため）
    english_fasttext_vectors = Vectors(name='./data/wiki-news-300d-1M.vec')

    TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)

    train_dl = torchtext.data.Iterator(train_ds, batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size, train=False, sort=False)

    return train_dl, val_dl, test_dl, TEXT


if __name__ == "__main__":
    print(tokenizer_with_preprocessing('I like cats.'))

    train_dl, val_dl, test_dl, TEXT = create_dataloader()
    batch = next(iter(val_dl))
    print(batch.Text)
    print(batch.Label)
    print(TEXT.vocab.vectors.shape)
