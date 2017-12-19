from gensim.models import Word2Vec
from corpus import Corpus

class Embeddings():
    def __init__(self, embeddings='Word2Vec'):
        self.model = self.init_model(embeddings)

    def init_model(self, embeddings):
        if embeddings == 'Word2Vec':
            return Word2Vec()

    def fit_corpus(self, text):
        corpus = Corpus(text)
        self.batches = [batch.split(' ') for batch in list(corpus.batches(dictionary=False))]

    def _build_vocab(self, batches):
        self.model.build_vocab(batches)

    def _train_model(self, batches, epochs=3):
        self.model.train(batches, total_examples=len(batches), epochs=epochs)

    def train(self):
        self._build_vocab(self.batches)
        self._train_model(self.batches)

    def vector(self, word):
        return self.model[word]

    def online_training(self):
        for batch in batches:
            model.build_vocab(batch, update=True)
            model.train(batch, total_examples=10, epochs=3)
