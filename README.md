# MultiLang

## Multi-label language detection using Gibbs sampling.


This implementation uses a Byte-Pair subword tokenization scheme which can be reliably trained over a multi-lingual corpus and is robust to out of distribution words. It is trained on an unmodified language detectiondataset, where in each document has only a single language label. However, the model learns to approximate the generative process of texts in a specific language and come inference time, the model uncovers all the training languages that a document is likely a mixture of, above a configurable threshold.

The generative process assumes that a given document is generated as a mixture of languages, each of which are generated from a mixture of words. The language-word conditional distribution is inferred using a maximum likelihood estimate on the training data, smoothed by an optional parameter GIBBS_BETA. The document-language conditional distribution on the other hand, is approximated using a Gibbs sampler, that runs for a configurable number of iterations and is smoothed by another optional parameter GIBBS_ALPHA.

The collapsed Gibbs sampling method has the following desirable qualities:

* **Multi Label:** The generative procedure is able to model a mixture of languages emanating from a set of vocabulary words. This allows us to easily handle multi language scenarios which would otherwise either not be possible or be very involved, requiring multi-label classification.

* **Robust to Sparse Data:** The gibbs sampling and the maximum likelihood procedure are not nearly as data hungry as a deep model would be. Additionally, configurable parameters such as GIBBS_ALPHA and GIBBS_BETA allow us to controllably smoothen the genrating distribution to mitigate the effects of sparsity.

* **Explainable:** Unlike deep learning models, attribution of words that influenced the language assignments the most, are readily accesible by definition. An inspection of the conditional probabilities would make it very clear.

This repository implements a learner for a generative model representing multiple languages over documents as published at https://www.aclweb.org/anthology/Q14-1003.pdf. Cite the original work as follows:

    @article{lui-etal-2014-automatic,
    title = "Automatic Detection and Language Identification of Multilingual Documents",
    author = "Lui, Marco  and
      Lau, Jey Han  and
      Baldwin, Timothy",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "2",
    year = "2014",
    url = "https://www.aclweb.org/anthology/Q14-1003",
    doi = "10.1162/tacl_a_00163",
    pages = "27--40",
    abstract = "Language identification is the task of automatically detecting the language(s) present in a document based on the content of the document. In this work, we address the problem of detecting documents that contain text from more than one language (multilingual documents). We introduce a method that is able to detect that a document is multilingual, identify the languages present, and estimate their relative proportions. We demonstrate the effectiveness of our method over synthetic data, as well as real-world multilingual documents collected from the web.",
    }

