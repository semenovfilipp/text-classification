# Text classification: Disaster Tweets

Решаю задачу бинарной классификации твитов из соревнования Kaggle [NLP Getting Started](https://www.kaggle.com/competitions/nlp-getting-started): определить, сообщает ли твит о реальной катастрофе (`target=1`) или нет (`target=0`). Метрика соревнования F1.

Сравниваю три подхода:

1. **Classic NLP**: разные варианты предобработки → BoW / TF-IDF / word2vec / GloVe → LogReg / LinearSVC / CatBoost.
2. **Эмбеддинги трансформеров + classic ML**: sentence-эмбеддинги из предобученных энкодеров (MiniLM, BGE, E5, Qwen3-Embedding) → LogReg / CatBoost.
3. **Fine-tune трансформера**: `cardiffnlp/twitter-roberta-base`, `vinai/bertweet-base` (GPU) и `distilroberta-base` (CPU-бейзлайн) через `transformers.Trainer`.

**Коротко о результатах** (val, сплит v1): classic F1 **0.7697** → E5-base + LogReg **0.8046** → fine-tune BERTweet **0.8053**. Полный прогон в Colab завершён: [86 экспериментов](results/metrics.csv), [сводная таблица](results/metrics_table.md). Отдельно сохранён [локальный CPU-прогон](results/metrics_table_local_cpu.md), где DistilRoBERTa получила F1 **0.8052**. Разница между лидерами мала; результаты относятся к одному сплиту с описанными ниже ограничениями.

---

## Данные и EDA

Ноутбук: [`notebooks/exploratory/01_eda.ipynb`](notebooks/exploratory/01_eda.ipynb)

- 7613 твитов в train и 3263 в test, колонки `id, keyword, location, text, target`.
- Классы 57% / 43%: дисбаланс умеренный, хватает стратифицированного сплита и F1.
- Тексты короткие: в среднем 101 символ (15 слов), максимум 157 символов и 85 токенов RoBERTa. `max_length=96` обрезки не даёт.
- `keyword` заполнен на 99% и сильно связан с меткой (`derailment`, `debris`, `wreckage` почти всегда 1; `aftershock`, `body bags` почти всегда 0). Один keyword даёт F1 ≈ 0.69. `location` грязный и заполнен на 67%, его не используем.
- **Шум разметки**: 18 текстов-дубликатов (55 строк) с разными метками. В сплите оставляем один экземпляр с меткой большинства, а тексты с ничьей удаляем. Это исключает пересечение точных исходных текстов между train и val; варианты с разными ссылками и упоминаниями могут остаться.
- Ссылки есть у 66% «катастрофных» твитов против 41% обычных, числа у 73% против 50%. Важен сам факт наличия, поэтому URL, @ и числа заменяем токенами, а не удаляем.
- Шум: HTML-сущности (`&amp;` ×342), битая кодировка (`Ûª`), хештеги, повторы символов.
- Train и test распределены одинаково.

## Валидация

`scripts/preprocess.py` один раз строит сплит `data/processed/v1_train_split.csv`: стратифицированный, 80/20, `seed=42`, после чистки дубликатов. Получается 5994 строки в train и 1499 в val. Все подходы учатся и оцениваются на одном и том же сплите. Test размечен только на Kaggle, по нему для каждого подхода сохраняется сабмит.

Метрики: F1 (основная), accuracy, precision, recall, ROC-AUC, `f1_tuned_thr` (F1 при оптимальном пороге, подобранном на val; оценка оптимистичная) и время обучения и инференса.

**Ограничение сплита v1.** После лёгкой замены URL и упоминаний 137 из 1499 текстов val совпадают с текстами train; после классической очистки `clean` — 149. Исходные тексты не пересекаются. Метрики могут быть оптимистичными, поэтому небольшие различия между моделями не доказывают превосходства одной из них. Для строгой оценки нужен отдельный сплит по группам нормализованных текстов с повторным обучением. В этой работе сохранён исходный v1 для сопоставимости выполненных экспериментов. [Проверка данных и результатов](results/validation_audit.md).

Основной F1 рассчитан при пороге 0.5 (для LinearSVC — 0). В четырёх исторических строках `f1_tuned_thr` ниже основного F1: сетка квантилей не включала стандартный порог. Значения архива сохранены; код исправлен для будущих запусков. Порог по val используется только для диагностики и не применяется к сабмитам.

## 1. Classic NLP

Конфиг: [`configs/classic.yaml`](configs/classic.yaml). Перебор задаётся секцией `experiments`.

**Варианты предобработки** (`src/data/preprocess.py`). Каждый сохраняется в отдельный файл `data/processed/v1_<name>.csv`.

| вариант | что делает |
|---|---|
| `raw` | токенизация как есть |
| `lower` | + нижний регистр |
| `clean_punct` | + `html.unescape`, URL/@/числа → `URL`/`USER`/`NUM`, `#` снят, повторы символов схлопнуты |
| `clean` | + удаление пунктуации |
| `clean_nostop` | + удаление стоп-слов NLTK (кроме `no/not/nor`) |
| `clean_stem` / `clean_lemma` | + стемминг Snowball / лемматизация WordNet |
| `clean_nostop_lemma` | стоп-слова + лемматизация |
| `clean_kw`, `clean_kw_stem` | keyword приклеен к тексту |

**Векторизация**: BoW (1–2-граммы), TF-IDF по словам, TF-IDF по символьным 2–5-граммам, TF-IDF word+char, word2vec (gensim, обучен на train) и предобученный GloVe-twitter-100. Для word2vec вектор твита считается как среднее векторов слов.

**Модели**: LogisticRegression, LinearSVC, CatBoost.

Результаты (лучшие строки, полная таблица в [`results/metrics_table_local_cpu.md`](results/metrics_table_local_cpu.md)). В локальном прогоне вариант `clean` назывался `clean_nopunct`, `clean_punct` назывался `clean`, а эмбеддинговые `clean` / `clean_kw` назывались `nokw_clean` / `kw_clean`:

| эксперимент | предобработка | векторизация | модель | F1 | ROC-AUC | train, с |
|---|---|---|---|---|---|---|
| предобработка | raw | TF-IDF word | LogReg | 0.728 | 0.845 | 0.5 |
| предобработка | lower | TF-IDF word | LogReg | 0.735 | 0.855 | 0.5 |
| предобработка | clean | TF-IDF word | LogReg | 0.748 | 0.861 | 0.5 |
| предобработка | clean_nostop | TF-IDF word | LogReg | 0.739 | 0.849 | 0.3 |
| предобработка | clean_stem | TF-IDF word | LogReg | 0.744 | 0.866 | 0.4 |
| предобработка | clean_lemma | TF-IDF word | LogReg | 0.744 | 0.866 | 0.4 |
| предобработка | clean_kw | TF-IDF word | LogReg | 0.756 | 0.864 | 0.4 |
| векторизация | clean_kw | BoW | LogReg | 0.744 | 0.853 | 0.5 |
| векторизация | clean_kw | TF-IDF char | LogReg | 0.763 | 0.869 | 1.5 |
| векторизация | clean_kw | word2vec (свой) | LogReg | 0.732 | 0.839 | 12.7 |
| векторизация | clean_kw | GloVe-twitter | LogReg | 0.752 | 0.860 | 0.8 |
| голова | clean_kw | TF-IDF word+char | **LogReg** | **0.770** | 0.872 | 1.8 |
| голова | clean_kw | TF-IDF word+char | LinearSVC | 0.751 | 0.867 | 2.4 |
| бустинг | clean_kw | TF-IDF 5k | CatBoost | 0.732 | 0.844 | 56.6 |
| бустинг | clean | GloVe-twitter | **CatBoost** | **0.770** | 0.871 | 18.7 |

Выводы:
- Предобработка нужна, но лёгкая. Чистка (регистр, токены вместо URL/@/чисел, снятие `#`, удаление пунктуации) даёт +2 п. F1 к `raw`.
- **Удаление стоп-слов вредит** (−1 п.): в коротких твитах служебные слова несут смысл («on fire» и «fire»).
- Стемминг и лемматизация почти ничего не дают: +0.5 п. к ROC-AUC, F1 не меняется.
- **keyword помогает**: +1–2 п. F1.
- Символьные n-граммы лучше словных: устойчивы к опечаткам и сленгу. Комбинация word+char даёт лучший результат.
- Собственный word2vec на 6k твитов слаб, предобученный на твитах GloVe заметно лучше.
- CatBoost на разреженном TF-IDF проигрывает линейной модели и учится в 30 раз дольше. На плотных векторах GloVe он сравнялся с LogReg.
- **Лучшая голова — LogReg**: при равном F1 она в 10 раз быстрее CatBoost. Её и берём на этап 2.

## 2. Эмбеддинги трансформера + LogReg

Конфиги: [`configs/embeddings.yaml`](configs/embeddings.yaml) (CPU: small и base), [`configs/embeddings_gpu.yaml`](configs/embeddings_gpu.yaml) (GPU: large и Qwen3).

**Как выбирал эмбеддер.** Смотрел на [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), вкладка English, колонка **Classification**: она ближе всего к нашей задаче (эмбеддинг → линейный классификатор). Отбирал открытые модели, которые помещаются в Colab T4 и разумно работают на CPU. Шли по возрастанию размера:

| модель | параметры | зачем взята |
|---|---|---|
| `all-MiniLM-L6-v2` | 22M | быстрый бейзлайн, самая популярная sentence-модель |
| `BAAI/bge-small-en-v1.5` | 33M | одна из сильнейших на MTEB Classification среди моделей до 50M |
| `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2` | 110M | топ среди base-моделей, по Classification близки к large при втрое меньшем размере |
| `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2` | 335M | large-версии тех же семейств (на GPU) |
| `Qwen/Qwen3-Embedding-0.6B` | 600M | одна из лучших моделей до 1B на MTEB, поддерживает instruction-промпт под задачу (на GPU) |

Для E5 нужен префикс `query: `, для Qwen3 instruction, это прописано в конфиге. Эмбеддинги нормализуются и кэшируются в `data/embeddings/`.

**А точно ли нужна предобработка?** Проверяем три варианта: `raw` (текст как есть), `clean` (только `html.unescape` и `@user`/`http` вместо упоминаний и ссылок) и `clean_kw` (то же плюс keyword). Стоп-слова, стемминг и лемматизация не применяются: энкодер обучен на естественном тексте и сам учитывает контекст.

Результаты (CPU):

| энкодер | предобработка | голова | F1 | ROC-AUC | encode train, с |
|---|---|---|---|---|---|
| MiniLM-L6 | clean | CatBoost | 0.782 | 0.876 | 55 |
| MiniLM-L6 | clean | LogReg | 0.771 | 0.879 | 55 |
| bge-small | clean | LogReg | 0.801 | 0.892 | 108 |
| bge-base | clean | LogReg | 0.792 | 0.889 | 359 |
| **e5-base-v2** | clean | **LogReg** | **0.805** | **0.893** | 384 |
| e5-base-v2 | clean_kw | LogReg | 0.793 | 0.890 | 420 |
| bge-base | clean_kw | LogReg | 0.787 | 0.885 | 371 |

Выводы:
- Даже самая маленькая bge-small (33M) с обычной LogReg обгоняет лучший классический пайплайн на 3 п. F1.
- e5-base лучшая, но выигрыш у bge-small (в 3,5 раза меньше) всего 0.4 п.
- **keyword к тексту вредит** (−1 п.): энкодер «размывает» смысл твита.
- CatBoost на эмбеддингах не лучше LogReg.

Завершённый Colab-прогон включает 7 энкодеров × 3 варианта текста × 3 классификатора. Лучшие результаты каждого энкодера:

| энкодер | предобработка | голова | F1 |
|---|---|---|---:|
| E5-base-v2 | clean | LogReg, C=1 или C=3 | 0.8046 |
| BGE-small-en-v1.5 | clean | LogReg, C=1 | 0.8010 |
| E5-large-v2 | raw | LogReg, C=3 | 0.7990 |
| Qwen3-Embedding-0.6B | raw или clean | LogReg, C=3 | 0.7973 |
| BGE-large-en-v1.5 | clean_kw | LogReg, C=3 | 0.7957 |
| BGE-base-en-v1.5 | clean_kw | LogReg, C=3 | 0.7908 |
| MiniLM-L6-v2 | clean | CatBoost | 0.7807 |

Увеличение размера энкодера не улучшило F1 на этом сплите: E5-base осталась лучшей среди эмбеддингов. Эти результаты не устанавливают общее превосходство base-моделей над large.

## 3. Fine-tune трансформера

Конфиги: [`configs/finetune.yaml`](configs/finetune.yaml), [`configs/finetune_bertweet.yaml`](configs/finetune_bertweet.yaml), [`configs/finetune_cpu.yaml`](configs/finetune_cpu.yaml).

**Почему эти модели:**
- **`cardiffnlp/twitter-roberta-base`** (основная). Это RoBERTa-base, дообученная на ~58M твитов ([TweetEval](https://arxiv.org/abs/2010.12421), Barbieri et al., 2020). Домен совпадает с нашим: сленг, хештеги, `@user`, опечатки. На TweetEval она обгоняет обычную RoBERTa-base. 125M параметров, на T4 обучается за несколько минут.
- **`vinai/bertweet-base`** (для сравнения). RoBERTa-архитектура, предобученная с нуля на 850M англоязычных твитов ([BERTweet](https://arxiv.org/abs/2005.10200), Nguyen et al., 2020). Это SOTA на твиттерных задачах из того же класса размеров.
- **`distilroberta-base`** (CPU-бейзлайн). 82M параметров, в два раза быстрее base, обучается на CPU за 30 минут.

**Какая нужна предобработка.** Нужна та же, что при предобучении модели, не больше:
- для twitter-roberta: `html.unescape`, `@user` вместо упоминаний и `http` вместо ссылок, ровно как в TweetEval;
- для bertweet: встроенная нормализация токенизатора (`normalization=True`: `@USER`, `HTTPURL`, эмодзи → текст);
- стоп-слова, пунктуацию и регистр не трогаем.

Параметры GPU-обучения: `lr=2e-5`, linear schedule с warmup 10%, `weight_decay=0.01`, batch 32, до 4 эпох с early stopping по F1 на val, fp16. `max_length=96` для Twitter-RoBERTa и 128 для BERTweet. CPU-бейзлайн DistilRoBERTa обучался до 2 эпох с `max_length=64`. Лоссы и метрики пишутся в TensorBoard (`logs/tensorboard/<model>`), лучший чекпоинт сохраняется в `models/finetune/<model>/best`.

| модель | устройство | F1 | ROC-AUC | train | infer val |
|---|---|---|---|---|---|
| distilroberta-base | CPU | 0.8052 | 0.8926 | 1801.80 с | 52.49 с |
| twitter-roberta-base | CUDA / Colab | 0.8025 | 0.8853 | 68.35 с | 0.93 с |
| bertweet-base | CUDA / Colab | 0.8053 | 0.8940 | 88.37 с | 0.85 с |

## Итоговая таблица

| прогон | подход / модель | F1 | accuracy | ROC-AUC | обучение, с | инференс val, с |
|---|---|---:|---:|---:|---:|---:|
| Colab, classic на CPU | GloVe-twitter + CatBoost | 0.7697 | 0.8132 | 0.8748 | 21.35 | 0.14 |
| Colab, CUDA-энкодер | E5-base + LogReg, C=3 | 0.8046 | 0.8412 | 0.8921 | 12.94 | 3.20 |
| Colab, CUDA-энкодер | BGE-small + LogReg, C=1 | 0.8010 | 0.8392 | 0.8917 | 3.88 | 1.00 |
| Colab, CUDA | BERTweet | 0.8053 | 0.8419 | 0.8940 | 88.37 | 0.85 |
| Локально, CPU | TF-IDF word+char + LogReg | 0.7700 | 0.8159 | 0.8723 | 1.77 | 0.37 |
| Локально, CPU | E5-base + LogReg, C=1 | 0.8046 | 0.8419 | 0.8933 | 383.93 | 92.09 |
| Локально, CPU | DistilRoBERTa | 0.8052 | 0.8399 | 0.8926 | 1801.80 | 52.49 |

Для эмбеддингов время обучения включает энкодинг train и обучение головы, инференс — энкодинг val и работу головы. Загрузка модели не входит в замер энкодинга. Кэш хранит время исходного вычисления, поэтому строка повторного запуска может содержать исторический замер. CPU- и CUDA-время не сравнивается напрямую; конкретная модель GPU в CSV не записана.

### Какая модель лучшая по качеству и по скорости?

- **По скорости**: TF-IDF word+char + LogReg. Обучение за 2 секунды, инференс меньше миллисекунды на твит, работает на любом CPU. F1 0.770.
- **Лучший баланс**: bge-small + LogReg. F1 0.801 (−0.4 п. от лучшего), энкодер на 33M параметров, около 17 мс на твит на CPU, обучение головы за секунды.
- **По качеству на v1**: BERTweet получила максимальный F1 0.8053. E5-base + LogReg (0.8046) и DistilRoBERTa (0.8052) практически на том же уровне. Разница 0.0007 F1 между BERTweet и E5-base не позволяет объявить устойчивого победителя по одному сплиту.

Сохранены пять сабмитов: classic, E5-base, E5-large, Twitter-RoBERTa и BERTweet. Все прошли проверку формата и ID; оценок Kaggle в полученных результатах нет. Сабмиты сформированы моделями, обученными на train-части v1, без последующего обучения на всей размеченной выборке.

### Как ещё улучшить результаты

- **Данные**: почистить шумную разметку (дубликаты с разными метками и явные ошибки, которые видны по ошибкам модели), добавить внешние размеченные данные по кризисным твитам (CrisisNLP, CrisisLex).
- **Порог**: подобрать порог под F1 на кросс-валидации (на val это даёт до +1 п.).
- **Валидация**: 5-fold CV вместо одного сплита и усреднение 5 моделей.
- **Ансамбли**: проверить стекинг TF-IDF-LogReg и трансформера по out-of-fold предсказаниям; прирост заранее не гарантирован.
- **Модели**: `bertweet-large`, `deberta-v3-large`, `roberta-large`, ансамбль нескольких seed.
- **Fine-tune**: подбор lr и числа эпох (Optuna), layer-wise lr decay, R-Drop, FGM-adversarial.
- **LLM**: эмбеддинги из 7B-эмбеддеров (e5-mistral, Qwen3-Embedding-8B) или zero/few-shot классификация LLM в качестве ещё одного признака.
- **Признаки**: подавать `keyword` трансформеру отдельным сегментом (`keyword </s> text`), а не склеивать с текстом.
- **Аугментации**: back-translation, EDA-аугментации, псевдо-разметка test.

---

## Структура репозитория

```
├── configs/                    # все гиперпараметры, пути и seed
│   ├── data.yaml               # общий: пути, сплит, версия данных, seed
│   ├── classic.yaml            # варианты предобработки × векторизаторы × модели
│   ├── embeddings.yaml         # энкодеры small/base (CPU)
│   ├── embeddings_gpu.yaml     # large и Qwen3 (GPU)
│   ├── finetune.yaml           # twitter-roberta-base
│   ├── finetune_bertweet.yaml
│   └── finetune_cpu.yaml       # distilroberta-base
├── notebooks/
│   ├── exploratory/01_eda.ipynb
│   └── experiments/colab_pipeline.ipynb   # полный прогон на GPU
├── scripts/                    # точки входа (CLI на click)
│   ├── preprocess.py           # сплит v1 и версии датасета
│   ├── train_classic.py
│   ├── train_embeddings.py
│   ├── finetune.py
│   └── evaluate.py             # сводная таблица метрик
├── src/
│   ├── data/                   # загрузка, сплит, предобработка
│   ├── features/               # векторизаторы, эмбеддинги
│   ├── models/                 # classic-модели, датасет для Trainer
│   └── utils/                  # конфиги, метрики, отчёт
├── results/                    # metrics*.csv, metrics_table*.md
├── requirements.txt
└── pyproject.toml
```

В git не попадают `data/`, `models/`, `logs/` и сабмиты.

Каждый запуск дописывает строку в `results/metrics.csv`: подход, конфигурация, метрики, время. Лучшая модель каждого скрипта сохраняется в `models/<approach>/` (`.joblib` для sklearn и CatBoost, HF-формат для fine-tune), сабмит для Kaggle сохраняется в `results/submissions/`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # только для локального CPU
pip install -r requirements.txt    # ставит и сам проект в editable-режиме
```

Данные. Нужен токен Kaggle в `~/.kaggle/kaggle.json` и принятые правила соревнования:

```bash
kaggle competitions download -c nlp-getting-started -p data/raw
unzip data/raw/nlp-getting-started.zip -d data/raw
```

GloVe для classic (если `gensim.downloader` недоступен, скрипт пытается скачать сам):

```bash
mkdir -p data/external
curl -L -o data/external/glove-twitter-100.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-100/glove-twitter-100.gz
```

## Запуск

```bash
python scripts/preprocess.py --config configs/classic.yaml
python scripts/train_classic.py --config configs/classic.yaml
python scripts/train_embeddings.py --config configs/embeddings.yaml
python scripts/train_embeddings.py --config configs/embeddings_gpu.yaml      # GPU
python scripts/finetune.py --config configs/finetune.yaml                    # GPU
python scripts/finetune.py --config configs/finetune_bertweet.yaml           # GPU
python scripts/evaluate.py
python scripts/audit_results.py --submissions results/submissions
python -m unittest discover -s tests
tensorboard --logdir logs/tensorboard
```

У каждого скрипта есть `--debug N` для быстрой проверки на подвыборке из N объектов. Метрики такого прогона пишутся в `results/metrics_debug.csv`, веса не сохраняются.

### Colab

[`notebooks/experiments/colab_pipeline.ipynb`](notebooks/experiments/colab_pipeline.ipynb) клонирует ветку, подключает Google Drive и запускает все скрипты по порядку на GPU. Токен Kaggle берётся из Colab Secrets (`KAGGLE_API_TOKEN` или `KAGGLE_USERNAME` + `KAGGLE_KEY`), в репозитории ключей нет. `data/`, `models/`, `logs/` и `results/` хранятся на Drive в `MyDrive/text-classification/`, поэтому не пропадают при отключении сессии. Если в Colab Secrets задан `GITHUB_TOKEN`, ноутбук сам коммитит `results/metrics.csv` и `results/metrics_table.md` в ветку.
