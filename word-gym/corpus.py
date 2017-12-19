from nltk.tokenize import sent_tokenize, word_tokenize

class Corpus:
    def __init__(self, corpus):
        self.corpus = corpus

    def batches(self, dictionary=False, batch_size=20):
        raw_text = self.corpus
        if dictionary:
            for i in self._split_sentences(raw_text):
                yield i
        else:
            while raw_text:
                yield raw_text[:batch_size]
                raw_text = raw_text[batch_size:]

    def _split_sentences(self, text, batch_size=20):
        batches = []
        for sentence in sent_tokenize(text):
            for i in self._find_neighbors(sentence):
                batches.append(i)
                if len(batches) > batch_size:
                    yield(batches)
                    batches = []
        yield(batches)


    def _find_neighbors(self, text_batch, window=3):
            words = word_tokenize(text_batch)
            for word_id, word in enumerate(words):
                lower_bound = word_id-window
                if word_id-window < 0:
                    lower_bound = 0
                d = {}
                d[word] = words[lower_bound:word_id] + words[word_id+1:word_id+window]
                yield(d)
