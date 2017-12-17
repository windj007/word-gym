from evalutor import Evaluator
from embeddings import Embeddings

class Pipeline():
    def __init__(self, text, model='Word2Vec'):
        self.model = Embeddings(model)
        self.model.fit_corpus(text)
        self.model.train()

    def evaluate(self, test='word-similarity', datasets=['wordsim353-rel']):
        evaluator = Evaluator(test='word-similarity', datasets=['wordsim353-rel'], metric='spearman')
        return evaluator.evaluate(self.model)
