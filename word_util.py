import os
import sentencepiece as spm
from gensim.models import Word2Vec
import pandas as pd
from data_util import make_input4tokenizer


def train_tokenizer(input_file_path, mode_file_path, vocab_size, model_type):
    templates = ' --input={} \
        --pad_id=0 \
        --bos_id=1 \
        --eos_id=2 \
        --unk_id=3 \
        --model_prefix={} \
        --vocab_size={} \
        --character_coverage=1.0 \
        --model_type={}'

    cmd = templates.format(input_file_path,
                mode_file_path,    # output model 이름
                vocab_size,# 작을수록 문장을 잘게 쪼갬
                model_type)# unigram (default), bpe, char

    spm.SentencePieceTrainer.Train(cmd)
    print("tokenizer {} is generated".format(mode_file_path))


def get_tokens(sp, sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = sp.EncodeAsPieces(sentence)
        new_tokens = []
        for token in tokens:
            token = token.replace("▁", "")
            if len(token) > 1:
                new_tokens.append(token)
        if len(new_tokens) > 1:
            tokenized_sentences.append(new_tokens)

    return tokenized_sentences


class string2vec():
    def __init__(self, train_data, size=200, window=5, min_count=2, workers=8, sg=1, hs=1):
        self.model = Word2Vec(train_data, size=size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs)

    def set_model(self, model_fn):
        self.model = Word2Vec.load(model_fn)

    def save_embeddings(self, emb_fn):
        word_vectors = self.model.wv

        vocabs = []
        vectors = []
        for key in word_vectors.vocab:
            vocabs.append(key)
            vectors.append(word_vectors[key])

        df = pd.DataFrame()
        df['voca'] = vocabs
        df['vector'] = vectors

        df.to_csv(emb_fn,index=False)

    def save_model(self,md_fn):
        self.model.save(md_fn)
        print("word embedding model {} is trained".format(md_fn))

    def show_similar_words(self,word, topn):
        print(self.model.most_similar(positive=[word], topn=topn))


if __name__ == '__main__':
    train_file_path = '../data/train.json'
    genre_file_path = '../data/genre_gn_pre.json'
    tokenize_input_file_path = '../data/tokenizer_input.txt'

    vocab_size = 12000
    method = 'bpe'
    tokenizer_file_path = 'model/tokenizer_{}_{}.model'

    sentences = make_input4tokenizer(train_file_path, genre_file_path, tokenize_input_file_path)

    if not os.path.exists(tokenizer_file_path):
        train_tokenizer(tokenize_input_file_path, tokenizer_file_path, vocab_size, method)

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_file_path)

    tokenized_sentences = get_tokens(sp, sentences)

    emb_fn = '../data/word_embeddings_{}_{}.csv'.format(method, vocab_size)
    model_fn = '../model/w2v_{}_{}.model'.format(method, vocab_size)

    model = string2vec(tokenized_sentences, size=200, window=5, min_count=1, workers=8, sg=1, hs=1)
    model.save_embeddings(emb_fn)
    model.save_model(model_fn)
