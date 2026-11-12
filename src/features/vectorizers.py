from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.utils.common import resolve


@lru_cache(maxsize=2)
def load_pretrained(path):
    from gensim.models import KeyedVectors

    src = resolve(path)
    cache = src.with_suffix(".kv")
    if cache.exists():
        return KeyedVectors.load(str(cache), mmap="r")
    if not src.exists():
        import gensim.downloader as api

        return api.load(src.stem)
    kv = KeyedVectors.load_word2vec_format(str(src), binary=False)
    kv.save(str(cache))
    return kv


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=30, sg=1, pretrained=None, tfidf_weighted=False, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.pretrained = pretrained
        self.tfidf_weighted = tfidf_weighted
        self.seed = seed

    def fit(self, X, y=None):
        sentences = [x.split() for x in X]
        if self.pretrained:
            self.kv_ = load_pretrained(self.pretrained)
        else:
            from gensim.models import Word2Vec

            model = Word2Vec(
                sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                epochs=self.epochs,
                seed=self.seed,
                workers=1,
            )
            self.kv_ = model.wv
        self.idf_ = None
        if self.tfidf_weighted:
            tf = TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False, token_pattern=None).fit(X)
            self.idf_ = dict(zip(tf.get_feature_names_out(), tf.idf_))
        return self

    def _embed(self, tokens):
        vecs, weights = [], []
        for t in tokens:
            key = t if t in self.kv_ else t.lower()
            if key in self.kv_:
                vecs.append(self.kv_[key])
                weights.append(self.idf_.get(t, 1.0) if self.idf_ else 1.0)
        if not vecs:
            return np.zeros(self.kv_.vector_size, dtype=np.float32)
        return np.average(np.vstack(vecs), axis=0, weights=weights)

    def transform(self, X):
        return np.vstack([self._embed(x.split()) for x in X])


def _text_vec(kind, params):
    params = dict(params or {})
    if "ngram_range" in params:
        params["ngram_range"] = tuple(params["ngram_range"])
    cls = CountVectorizer if kind == "bow" else TfidfVectorizer
    return cls(token_pattern=r"(?u)\S+" if params.get("analyzer", "word") == "word" else None, lowercase=False, **params)


def build_vectorizer(spec):
    kind = spec["type"]
    params = spec.get("params", {})
    if kind in ("bow", "tfidf"):
        return _text_vec(kind, params)
    if kind == "tfidf_union":
        return FeatureUnion([(name, _text_vec("tfidf", p)) for name, p in params.items()])
    if kind == "word2vec":
        return Word2VecVectorizer(**params)
    raise ValueError(kind)
