# Arabic Dialect Geographic Identifier

A complete pipeline for identifying 25 regional Arabic dialects plus Modern Standard Arabic (MSA) from raw text and visualizing predictions on an interactive map.

## Overview of the Processing Steps

1. **Data Loading**  
   • Read parallel MADAR corpus covering 26 categories (25 city dialects + MSA).  
   • Assign dialect and region labels for each sentence.

2. **Preprocessing**  
   A custom dialect-aware preprocessor applies:
   - CODA-compliant character normalization (alef variants, taa marbuta)  
   - Diacritic removal, punctuation cleanup, whitespace normalization  
   - Detection and tokenization of common multi-word dialectal expressions  
   - City-specific phonological mapping based on CAPHI guidelines  
   - Morphological handling of prefixes, suffixes, and circumfix negation  
   - Optional lexical normalization using a MADAR lexicon

3. **Word Embeddings**  
   Experiments include:
   - AraVec CBOW 300‑dim models (wiki & Twitter variants)  
   - FastText Arabic vectors  
   - Custom Word2Vec trained on the MADAR text

   *Example loading code:*  
   ```python
   from gensim.models import Word2Vec, KeyedVectors
   wiki = Word2Vec.load('cbow_300_wiki.mdl').wv
   twitter = Word2Vec.load('cbow_300_twitter.mdl').wv
   fasttext = KeyedVectors.load_word2vec_format('cc.ar.300.vec')
   ```

4. **Model Training**  
   **BiLSTM‑CNN with Attention**  
   - Embedding layer (pretrained or custom)  
   - Bidirectional LSTM for sequence representation  
   - Multi‑scale 1D CNNs to capture n‑gram patterns  
   - Self‑attention over combined CNN outputs  
   - A specialized feature layer injecting dialect bias  
   - Dense layers to classify into 26 dialect categories

   **Transformer Fine‑tuning**  
   - MARBERT for multi‑dialect classification  
   - AraBERT for MSA or embedding experiments  
   - Support for other Arabic BERT variants via Hugging Face

   *Example transformer load:*  
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERT')
   model = AutoModelForSequenceClassification.from_pretrained(
       'UBC-NLP/MARBERT', num_labels=26)
   ```

5. **Evaluation**  
   Training and validation track accuracy and F1 (macro and weighted).  
   Final performance examples:  
   | Model               | Accuracy | F1 (macro) | F1 (weighted) |  
   |---------------------|:--------:|:----------:|:-------------:|  
   | BiLSTM‑CNN (FastText)|   0.76   |    0.74    |     0.75      |  
   | BiLSTM‑CNN (AraVec)  |   0.78   |    0.77    |     0.77      |  
   | MARBERT             |   0.82   |    0.81    |     0.82      |  
   | Ensemble            |   0.84   |    0.83    |     0.83      |

6. **Web Application**  
   A Flask service handles:
   - Text submission and model selection  
   - On‑the‑fly preprocessing and prediction  
   - Generation of a Folium heatmap or point map
   - A responsive front end for real‑time interaction

## Usage Instructions

1. Install prerequisites:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the MADAR corpus is placed in a `MADAR_Corpus/` folder.  
3. Start the server:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser.  
5. Enter Arabic text, choose a model and map type, then click **Identify Dialect**.

## Screenshots

**1. Input Form & Button**  
![Enter text and click button](screenshots/input_example.png)

**2. Predictions & Map**  
![Dialect probabilities and geographic heatmap](screenshots/output_map.png)

## Citation

If you use this work, please cite the MADAR corpus:

> Bouamor et al., “MADAR: A Parallel Multi‑dialect Corpus of Arabic,” 2021.

*Last updated: May 1, 2025.*
