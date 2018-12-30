
# Sentence Embedding


With a language model constructed from a sentence corpus we can give a representation for each sentence (sentence embedding) or, similarly, form a vector encoding for it (sentence encoder) and maybe generalize to *not seen sentences*. That sentence embedding can be lastly used in NLP and ML model for a specific task.

The state-of-art language models are based on pre-trained word embedding with predictive perspective (word2vec: CBOW and skip-gram), or coocurrences and count based (GloVe) or with more rich sub-word information (FastText), convolutional based (spaCy) or sequential language information based (ELMo), or other publication and DIY models (combination of convolutional, sequential blocks; or character based; or more complex ones with siamese, encoder-decoder, etc.). Also, could be usefull to compare and try distributed representations (Paragraph Vector, Doc2vec) and bayesian models with topics (LDA, LDA2Vec) more used directly for documents representation.

2018 research points out perspectives of pre-trained and Universal language models (Google's Universal Sentence Encoder and BERT), fine-tunning models for text classification (ULMFit by Jeremy Howard and Sebastian Ruder) as evolution to the approach of transfer learning in natural language processing.

[Last Update Date: 30 December 2018]


## Universal Sentence Encoder

Google Research publication and libraries implementation in TensorFlow Hub. As a result the pre-trained language models give a 512 size vector in the sentence embedding space. To mesuare similarity, it is enough to do inner product between sentences vectors. If needed, there is an introduction to TensorFlow Embeddings in [TensorFlow Guide](https://www.tensorflow.org/guide/embedding).

Two pre-trained models available only in English: Transformer and Deep Averaging Network (DAN). 

*Surprise!* There is one implementation available for Spanish. See more info in libraries, more languages are also supported with perspective on shared implementation with English pre-trained models (English-Spanish, English-French, English-German).

Repository file proofs: univ-sent-encoder.py


### Code Requirements

For English version, tested and code working: Python (3.5.1), TensorFlow (1.7), TensorFlow Hub.
``` 
pip3 install tensorflow==1.7
pip3 install tensorflow-hub
pip3 install seaborn
```

For Spanish version: previous English configuration and:
```
pip3 install sentencepiece
pip3 install tf-sentencepiece
```
Actual status: *not working on my code* with Python (3.5.1), TensorFlow (1.7). [WIP]


### Code References and Further Reading

1. [TensorFlow Hub Libraries](https://tfhub.dev/s?q=universal%20sentence%20encoder)

2. [Google Colab Notebook](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb)

3. Cer, D., Yang, Y., Kong, S. Y., Hua, N., Limtiaco, N., John, R. S., ... & Sung, Y. H. (2018). Universal sentence encoder. arXiv preprint arXiv:1803.11175. [See in arXiv](https://arxiv.org/pdf/1803.11175.pdf)

4. Chidambaram, M., Yang, Y., Cer, D., Yuan, S., Sung, Y. H., Strope, B., & Kurzweil, R. (2018). Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model. arXiv preprint arXiv:1810.12836. [See in arXiv](https://arxiv.org/pdf/1810.12836.pdf)

5. Tutorial on Universal Sentence Encoder and proofs with Avengers: Infinity War script. [Learn OpenCV: Universal Sentence Encoder](https://www.learnopencv.com/universal-sentence-encoder/)

6. Tutorial by Chengwei Zhang: Keras + Universal Sentence Encoder = Transfer Learning for text data. Also, so useful image to compare between Transformer and DAN models! [DLology Practical Deep Learning](https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/)


## Useful Sentence Datasets to proof

1. For Semantic Textual Similarity (STS) NLP task: [Semantic Textual Similarity Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)

2. For Sentence Classification, for example questions: [Experimental Data for Question Classification](http://cogcomp.org/Data/QA/QC/)


## References and Further Reading

1. More related info with STS and Sentence Embedding in Facebook Research work: [SentEval: evaluation toolkit for sentence embeddings](https://github.com/facebookresearch/SentEval)

2. NLP Progress by Sebastian Ruder: [Semantic Textual Similarity](https://nlpprogress.com/english/semantic_textual_similarity.html)