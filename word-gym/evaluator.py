from os import path
from pandas import DataFrame
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


class Evaluator():
    def __init__(self, test='word_similarity', datasets=['wordsim353-rel'], metric='spearman'):
        self.datasets_dir = 'datasets'

        self.test = test
        self.datasets = [DataFrame.from_csv(path.join(self.datasets_dir, test, '{}.csv'.format(dataset))).dropna() for dataset in datasets]
        self.metric = metric

    def evaluate(self, model):
        similarities = {'model':[], 'human':[]}
        for dataset in self.datasets:
            for index, row in dataset.iterrows():
                try:
                    similarities['model'].append(1 - cosine(model.vector(row['word1']), model.vector(row['word2'])))
                    similarities['human'].append(row['similarity'])
                except KeyError:
                    pass
        if self.metric == 'spearman':
            return spearmanr(similarities['model'], similarities['human'])
