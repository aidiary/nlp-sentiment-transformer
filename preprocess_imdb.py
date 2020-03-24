import glob
import os
import io


def make_train_data():
    f = open('./data/IMDb_train.tsv', 'w')

    path = './data/aclImdb/train/pos/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding='utf-8') as ff:
            text = ff.readline()
            text = text.replace('\t', '')
            text = text + '\t' + '1' + '\t' + '\n'
            f.write(text)

    path = './data/aclImdb/train/neg/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding='utf-8') as ff:
            text = ff.readline()
            text = text.replace('\t', '')
            text = text + '\t' + '0' + '\t' + '\n'
            f.write(text)

    f.close()


def make_test_data():
    f = open('./data/IMDb_test.tsv', 'w')

    path = './data/aclImdb/test/pos/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding='utf-8') as ff:
            text = ff.readline()
            text = text.replace('\t', '')
            text = text + '\t' + '1' + '\t' + '\n'
            f.write(text)

    path = './data/aclImdb/test/neg/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding='utf-8') as ff:
            text = ff.readline()
            text = text.replace('\t', '')
            text = text + '\t' + '0' + '\t' + '\n'
            f.write(text)

    f.close()


if __name__ == "__main__":
    make_train_data()
    make_test_data()
