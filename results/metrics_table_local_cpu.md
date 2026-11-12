## Лучшая модель каждого подхода

| approach | preprocessing | vectorizer | model | f1 | accuracy | precision | recall | roc_auc | f1_tuned_thr | train_time_s | infer_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finetune | light_clean | - | distilroberta-base | 0.8052 | 0.8399 | 0.835 | 0.7774 | 0.8926 | 0.8062 | 1801.8 | 52.49 |
| embeddings | nokw_clean | intfloat/e5-base-v2 | logreg | 0.8046 | 0.8419 | 0.8487 | 0.7649 | 0.8933 | 0.8033 | 383.93 | 92.09 |
| classic | clean | w2v_glove_twitter | catboost | 0.7701 | 0.8172 | 0.8285 | 0.7194 | 0.8709 | 0.7695 | 18.71 | 0.25 |

## classic: top-10 по F1

| approach | preprocessing | vectorizer | model | f1 | accuracy | precision | recall | roc_auc | f1_tuned_thr | train_time_s | infer_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classic | clean | w2v_glove_twitter | catboost | 0.7701 | 0.8172 | 0.8285 | 0.7194 | 0.8709 | 0.7695 | 18.71 | 0.25 |
| classic | clean_kw | tfidf_word_char | logreg | 0.77 | 0.8159 | 0.8221 | 0.7241 | 0.8723 | 0.7775 | 1.77 | 0.37 |
| classic | clean_kw | w2v_glove_twitter | catboost | 0.7697 | 0.8132 | 0.8097 | 0.7335 | 0.8748 | 0.7706 | 16.64 | 0.21 |
| classic | clean_kw | tfidf_word_char | logreg_balanced | 0.763 | 0.8019 | 0.7772 | 0.7492 | 0.8693 | 0.7655 | 1.92 | 0.55 |
| classic | clean_kw | tfidf_char | logreg_c3 | 0.7625 | 0.8092 | 0.811 | 0.7194 | 0.869 | 0.7678 | 1.5 | 0.31 |
| classic | clean_kw | tfidf_word | logreg_balanced | 0.7602 | 0.7992 | 0.7731 | 0.7476 | 0.8639 | 0.765 | 0.3 | 0.05 |
| classic | clean_kw | tfidf_word_uni | logreg_c3 | 0.7577 | 0.8059 | 0.8082 | 0.7132 | 0.8668 | 0.7674 | 0.18 | 0.02 |
| classic | clean | tfidf_word_char | logreg_c3 | 0.7577 | 0.8045 | 0.8021 | 0.7179 | 0.867 | 0.7631 | 1.84 | 0.36 |
| classic | clean_kw | tfidf_word | logreg_c3 | 0.7564 | 0.8045 | 0.8053 | 0.7132 | 0.8644 | 0.7651 | 0.43 | 0.06 |
| classic | clean_kw | tfidf_word_char | logreg_c3 | 0.7558 | 0.8025 | 0.7979 | 0.7179 | 0.87 | 0.7685 | 2.06 | 0.36 |

## embeddings: top-10 по F1

| approach | preprocessing | vectorizer | model | f1 | accuracy | precision | recall | roc_auc | f1_tuned_thr | train_time_s | infer_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| embeddings | nokw_clean | intfloat/e5-base-v2 | logreg_c3 | 0.8046 | 0.8412 | 0.8448 | 0.768 | 0.8921 | 0.805 | 383.94 | 92.09 |
| embeddings | nokw_clean | intfloat/e5-base-v2 | logreg | 0.8046 | 0.8419 | 0.8487 | 0.7649 | 0.8933 | 0.8033 | 383.93 | 92.09 |
| embeddings | nokw_clean | BAAI/bge-small-en-v1.5 | logreg | 0.801 | 0.8392 | 0.8464 | 0.7602 | 0.8917 | 0.8033 | 107.85 | 26.27 |
| embeddings | nokw_clean | BAAI/bge-small-en-v1.5 | logreg_c3 | 0.7993 | 0.8379 | 0.8447 | 0.7586 | 0.8901 | 0.8016 | 107.84 | 26.27 |
| embeddings | nokw_clean | BAAI/bge-base-en-v1.5 | catboost | 0.7967 | 0.8359 | 0.8427 | 0.7555 | 0.8844 | 0.7967 | 476.12 | 105.25 |
| embeddings | nokw_clean | intfloat/e5-base-v2 | logreg_c10 | 0.7955 | 0.8319 | 0.8249 | 0.768 | 0.8871 | 0.7997 | 383.95 | 92.09 |
| embeddings | nokw_clean | BAAI/bge-small-en-v1.5 | logreg_c10 | 0.7931 | 0.8319 | 0.8328 | 0.7571 | 0.8858 | 0.7967 | 107.85 | 26.27 |
| embeddings | kw_clean | intfloat/e5-base-v2 | logreg_c3 | 0.7931 | 0.8319 | 0.8328 | 0.7571 | 0.8896 | 0.7952 | 420.07 | 99.0 |
| embeddings | nokw_clean | BAAI/bge-base-en-v1.5 | logreg_c3 | 0.7918 | 0.8312 | 0.8336 | 0.7539 | 0.8889 | 0.7965 | 359.14 | 105.22 |
| embeddings | kw_clean | intfloat/e5-base-v2 | logreg_c10 | 0.7908 | 0.8306 | 0.8333 | 0.7524 | 0.8859 | 0.7968 | 420.09 | 99.0 |

## finetune: top-10 по F1

| approach | preprocessing | vectorizer | model | f1 | accuracy | precision | recall | roc_auc | f1_tuned_thr | train_time_s | infer_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finetune | light_clean | - | distilroberta-base | 0.8052 | 0.8399 | 0.835 | 0.7774 | 0.8926 | 0.8062 | 1801.8 | 52.49 |
