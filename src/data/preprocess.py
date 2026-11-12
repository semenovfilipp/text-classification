import html
import re
from functools import lru_cache

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?|[#@]?\w+|[^\w\s]")
NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
REPEAT_RE = re.compile(r"(.)\1{2,}")
SPACE_RE = re.compile(r"\s+")


def _nltk(resource, path):
    import nltk

    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def stopwords_set():
    _nltk("stopwords", "corpora/stopwords")
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english")) - {"no", "not", "nor"}


@lru_cache(maxsize=1)
def stemmer():
    from nltk.stem import SnowballStemmer

    return SnowballStemmer("english")


@lru_cache(maxsize=1)
def lemmatizer():
    _nltk("wordnet", "corpora/wordnet")
    _nltk("omw-1.4", "corpora/omw-1.4")
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


def _replace(regex, text, mode, token):
    if mode == "remove":
        return regex.sub(" ", text)
    if mode == "token":
        return regex.sub(f" {token} ", text)
    return text


def tokenize(text, opts):
    opts = opts or {}
    if opts.get("html_unescape", True):
        text = html.unescape(text)
    text = _replace(URL_RE, text, opts.get("urls", "keep"), "URL")
    text = _replace(MENTION_RE, text, opts.get("mentions", "keep"), "USER")
    text = _replace(NUM_RE, text, opts.get("numbers", "keep"), "NUM")
    if opts.get("ascii_only", False):
        text = NON_ASCII_RE.sub(" ", text)
    if opts.get("squeeze_repeats", False):
        text = REPEAT_RE.sub(r"\1\1", text)
    if opts.get("lowercase", False):
        text = text.lower()
    tokens = TOKEN_RE.findall(text)
    if opts.get("hashtags", "keep") == "strip":
        tokens = [t.lstrip("#") for t in tokens]
    if opts.get("remove_punct", False):
        tokens = [t for t in tokens if re.search(r"\w", t)]
    if opts.get("remove_stopwords", False):
        sw = stopwords_set()
        tokens = [t for t in tokens if t.lower() not in sw]
    norm = opts.get("normalize", "none")
    if norm == "stem":
        st = stemmer()
        tokens = [st.stem(t) if t not in ("URL", "USER", "NUM") else t for t in tokens]
    elif norm == "lemma":
        lm = lemmatizer()
        tokens = [lm.lemmatize(lm.lemmatize(t), pos="v") if len(t) > 3 else t for t in tokens]
    min_len = opts.get("min_token_len", 1)
    return [t for t in tokens if len(t) >= min_len]


def preprocess(texts, opts):
    return [" ".join(tokenize(t, opts)) for t in texts]


def light_clean(text, url_token="http", user_token="@user"):
    text = html.unescape(text)
    text = URL_RE.sub(url_token, text)
    text = MENTION_RE.sub(user_token, text)
    return SPACE_RE.sub(" ", text).strip()


def light_preprocess(texts, enabled=True, **kw):
    return [light_clean(t, **kw) if enabled else t for t in texts]
