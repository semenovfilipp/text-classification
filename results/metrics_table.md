## Лучшая модель каждого подхода

| approach | preprocessing | vectorizer | model | f1 | accuracy | precision | recall | roc_auc | f1_tuned_thr | train_time_s | infer_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finetune | tokenizer_norm | - | vinai/bertweet-base | 0.8053 | 0.8419 | 0.8463 | 0.768 | 0.894 | 0.8061 | 88.37 | 0.85 |
| embeddings | clean | intfloat/e5-base-v2 | logreg_c3 | 0.8046 | 0.8412 | 0.8448 | 0.768 | 0.8921 | 0.805 | 12.94 | 3.2 |
| classic | clean_kw | w2v_glove_twitter | catboost | 0.7697 | 0.8132 | 0.8097 | 0.7335 | 0.8748 | 0.7706 | 21.35 | 0.14 |

## classic: top-15 по F1

| approach | preprocessing | vectorizer | model | f1 | accuracy | precision | recall | roc_auc | f1_tuned_thr | train_time_s | infer_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classic | clean_kw | w2v_glove_twitter | catboost | 0.7697 | 0.8132 | 0.8097 | 0.7335 | 0.8748 | 0.7706 | 21.35 | 0.14 |
| classic | clean_kw | tfidf_word_char | logreg_balanced | 0.763 | 0.8019 | 0.7772 | 0.7492 | 0.8693 | 0.7655 | 3.35 | 0.47 |
| classic | clean_kw | tfidf_char | logreg | 0.7625 | 0.8092 | 0.811 | 0.7194 | 0.869 | 0.7678 | 1.2 | 0.29 |
| classic | clean | w2v_glove_twitter | catboost | 0.76 | 0.8079 | 0.8114 | 0.7147 | 0.8727 | 0.7716 | 22.55 | 0.13 |
| classic | clean_kw | tfidf_word | logreg | 0.7564 | 0.8045 | 0.8053 | 0.7132 | 0.8644 | 0.7651 | 0.33 | 0.09 |
| classic | clean_kw | tfidf_word_char | logreg | 0.7558 | 0.8025 | 0.7979 | 0.7179 | 0.87 | 0.7685 | 2.97 | 1.48 |
| classic | clean_kw_stem | tfidf_word | logreg | 0.755 | 0.8025 | 0.8 | 0.7147 | 0.8691 | 0.7639 | 0.33 | 0.07 |
| classic | clean_kw | w2v_glove_twitter | logreg | 0.7521 | 0.7999 | 0.7955 | 0.7132 | 0.8598 | 0.7621 | 166.57 | 0.33 |
| classic | clean_kw | tfidf_word_char | linear_svc | 0.7512 | 0.7999 | 0.7975 | 0.71 | 0.8673 | 0.7655 | 1.26 | 0.24 |
| classic | clean | tfidf_word | logreg | 0.7481 | 0.7992 | 0.8025 | 0.7006 | 0.8608 | 0.7556 | 2.33 | 0.13 |
| classic | clean_stem | tfidf_word | logreg | 0.7443 | 0.7965 | 0.8 | 0.6959 | 0.8662 | 0.7551 | 0.57 | 0.12 |
| classic | clean_lemma | tfidf_word | logreg | 0.744 | 0.7952 | 0.795 | 0.6991 | 0.8655 | 0.7596 | 0.59 | 0.12 |
| classic | clean_nostop_lemma | tfidf_word | logreg | 0.7436 | 0.7939 | 0.7901 | 0.7022 | 0.8542 | 0.7476 | 0.28 | 0.04 |
| classic | clean_kw | bow | logreg | 0.7435 | 0.7905 | 0.7765 | 0.7132 | 0.8525 | 0.7492 | 0.94 | 0.07 |
| classic | clean_punct | tfidf_word | logreg | 0.7391 | 0.7919 | 0.7921 | 0.6928 | 0.8563 | 0.7473 | 1.23 | 0.1 |

## embeddings: top-15 по F1

| approach | preprocessing | vectorizer | model | f1 | accuracy | precision | recall | roc_auc | f1_tuned_thr | train_time_s | infer_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| embeddings | clean | intfloat/e5-base-v2 | logreg_c3 | 0.8046 | 0.8412 | 0.8448 | 0.768 | 0.8921 | 0.805 | 12.94 | 3.2 |
| embeddings | clean | intfloat/e5-base-v2 | logreg | 0.8046 | 0.8419 | 0.8487 | 0.7649 | 0.8933 | 0.8033 | 13.02 | 3.21 |
| embeddings | clean | BAAI/bge-small-en-v1.5 | logreg | 0.801 | 0.8392 | 0.8464 | 0.7602 | 0.8917 | 0.8033 | 3.88 | 1.0 |
| embeddings | raw | intfloat/e5-base-v2 | logreg_c3 | 0.8007 | 0.8372 | 0.8362 | 0.768 | 0.8922 | 0.8049 | 17.85 | 4.47 |
| embeddings | raw | BAAI/bge-small-en-v1.5 | logreg_c3 | 0.8003 | 0.8372 | 0.8373 | 0.7665 | 0.8907 | 0.8061 | 5.87 | 1.5 |
| embeddings | clean | BAAI/bge-small-en-v1.5 | catboost | 0.8 | 0.8392 | 0.8501 | 0.7555 | 0.8833 | 0.8017 | 84.93 | 1.02 |
| embeddings | clean | BAAI/bge-small-en-v1.5 | logreg_c3 | 0.7993 | 0.8379 | 0.8447 | 0.7586 | 0.8901 | 0.8016 | 3.94 | 1.0 |
| embeddings | raw | BAAI/bge-small-en-v1.5 | logreg | 0.799 | 0.8366 | 0.8382 | 0.7633 | 0.8917 | 0.8033 | 5.77 | 1.5 |
| embeddings | raw | intfloat/e5-large-v2 | logreg_c3 | 0.799 | 0.8359 | 0.8345 | 0.7665 | 0.8906 | 0.7997 | 57.28 | 14.07 |
| embeddings | clean | intfloat/e5-large-v2 | logreg | 0.798 | 0.8366 | 0.8417 | 0.7586 | 0.8918 | 0.7984 | 42.08 | 10.04 |
| embeddings | raw | intfloat/e5-large-v2 | logreg | 0.798 | 0.8359 | 0.8379 | 0.7618 | 0.8909 | 0.7968 | 57.17 | 14.06 |
| embeddings | clean | Qwen/Qwen3-Embedding-0.6B | logreg_c3 | 0.7973 | 0.8372 | 0.8481 | 0.7524 | 0.888 | 0.7997 | 125.52 | 31.29 |
| embeddings | raw | Qwen/Qwen3-Embedding-0.6B | logreg_c3 | 0.7973 | 0.8372 | 0.8481 | 0.7524 | 0.8905 | 0.7964 | 153.71 | 38.34 |
| embeddings | raw | intfloat/e5-base-v2 | logreg | 0.7964 | 0.8352 | 0.84 | 0.7571 | 0.8935 | 0.8029 | 17.79 | 4.47 |
| embeddings | clean_kw | BAAI/bge-large-en-v1.5 | logreg_c3 | 0.7957 | 0.8346 | 0.8385 | 0.7571 | 0.8858 | 0.7981 | 41.8 | 10.13 |

## finetune: top-15 по F1

| approach | preprocessing | vectorizer | model | f1 | accuracy | precision | recall | roc_auc | f1_tuned_thr | train_time_s | infer_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finetune | tokenizer_norm | - | vinai/bertweet-base | 0.8053 | 0.8419 | 0.8463 | 0.768 | 0.894 | 0.8061 | 88.37 | 0.85 |
| finetune | light_clean | - | cardiffnlp/twitter-roberta-base | 0.8025 | 0.8326 | 0.8057 | 0.7994 | 0.8853 | 0.8032 | 68.35 | 0.93 |
