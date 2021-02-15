# Configuration ingestion.
from Config import Configuration

# Tokenizer utilities.
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# File utilities.
import pathlib
import pandas
import html
import ftfy
import bs4
import re
from typing import List, Dict

# History utilities.
import numpy
import copy
import itertools


class MultiLangLearner:
    """
    Learner class to learn a generative model representing multiple languages over documents as published at
    https://www.aclweb.org/anthology/Q14-1003.pdf.

    This implementation uses a Byte-Pair subword tokenization scheme which can be reliably trained over a
    multi-lingual corpus and is robust to out of distribution words. It is trained on an unmodified language detection
    dataset, where in each document has only a single language label. However, the model learns to approximate the
    generative process of texts in a specific language and come inference time, the model uncovers all the training
    languages that a document is likely a mixture of, above a configurable threshold.

    The generative process assumes that a given document is generated as a mixture of languages, each of which are
    generated from a mixture of words. The language-word conditional distribution is inferred using a maximum
    likelihood estimate on the training data, smoothed by an optional parameter GIBBS_BETA. The document-language
    conditional distribution on the other hand, is approximated using a Gibbs sampler, that runs for a configurable
    number of iterations and is smoothed by another optional parameter GIBBS_ALPHA.
    """

    def __init__(self):
        self.configuration = Configuration()
        self._train_tokenizer(self.configuration)
        self.parser = Tokenizer.from_file(self.configuration.TOKENIZER_PATH)
        self.frequency_history = copy.deepcopy(self._process_corpus(self.configuration, self.parser))

    @classmethod
    def _train_tokenizer(cls,
                         config: Configuration) -> None:
        """
        Trains the Byte-Pair Encoding Tokenizer and saves it in the configured path.

        Parameters
        ----------
        config: Configuration
            The configuration object.
        """

        path_gen = pathlib.Path(config.CORPUS_DIRECTORY)
        parser = Tokenizer(BPE())
        parse_trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                                   vocab_size=config.VOCAB_SIZE,
                                   min_frequency=config.MINIMUM_FREQUENCY)
        parser.train(files=list(path_gen.glob('**/*')), trainer=parse_trainer)
        parser.save(config.TOKENIZER_PATH)

    @classmethod
    def _process_document(cls,
                          config: Configuration,
                          doc_path: str,
                          parser: Tokenizer) -> List[int]:
        """
        Preprocesses the document text filtering out newlines, excess whitespaces, unicode artifacts and HTML tags.
        Finally tokenizes the text into the Byte-Pair feature space.

        Parameters
        ----------
        config: Configuration
            The configuration object.
        doc_path: str
            The file path to the document.
        parser: Tokenizer
            The pre-trained Byte-Pair tokenizer.

        Returns
        -------
        encode_list: list[int]
            Subword tokens of the cleaned document, featurized in the Byte-Pair space.
        """

        file = open(doc_path, "r")
        creative = file.read()
        file.close()
        creative = html.unescape(creative)
        creative = ftfy.fix_text(creative)
        creative = bs4.BeautifulSoup(creative, 'html.parser').get_text()
        creative = re.sub(r"(?x)\b(?=\w*\d)\w+\s*", "", creative, 0, re.UNICODE)
        creative = re.sub(r'\(([^)]+)\)', '', creative, 0, re.UNICODE)
        creative = config.RE_SHORT_URL.sub("Website", config.RE_URL.sub("Website", creative))
        creative = config.RE_EMAIL.sub("Email", creative)
        creative = config.RE_ZWSP.sub("", creative)
        creative = config.RE_LINEBREAK.sub(r"\n", creative)
        creative = config.RE_NONBREAKING_SPACE.sub(" ", creative)
        creative = creative.strip()

        return parser.encode(creative).ids

    @classmethod
    def _process_corpus(cls,
                        config: Configuration,
                        parser: Tokenizer) -> numpy.ndarray:
        """
        Iterates over all documents in the training corpus, tokenizes them using a sub-routine and stores the
        smoothed maximum-likelihood estimates.

        Parameters
        ----------
        config: Configuration
            The configuration object.
        parser: Tokenizer
            The configuration object.

        Returns
        -------
        frequency_history: numpy.ndarray
            The smoothed language-word co-occurence matrix.
        """

        lang_set = config.CORPUS_LANGUAGES
        corpus_map = pandas.read_csv(config.CORPUS_MAP_PATH, header=0)
        frequency_history = numpy.full((parser.get_vocab_size(), len(lang_set)), config.GIBBS_BETA)
        for row in corpus_map.iterrows():
            history_update = copy.deepcopy(cls._process_document(config, row[1]["Path"], parser))
            history_target = row[1]["Language"]
            if history_target not in config.CORPUS_LANGUAGES:
                raise ValueError("Training set contains language not in configuration")
            for token in history_update:
                column_index = lang_set.index(history_target)
                frequency_history[token, column_index] = frequency_history[token, column_index] + 1

        return frequency_history

    @classmethod
    def _gibbs_sampler(cls,
                       config: Configuration,
                       frequency_history: numpy.ndarray,
                       encoding: List[int]) -> List[numpy.ndarray]:
        """
        Runs the smoothed Gibbs sampler to approximate the language-word generative distribution, over a configurable
        number of iterations.

        Parameters
        ----------
        config: Configuration
            The configuration object.
        frequency_history: numpy.ndarray
            The smoothed language-word co-occurence matrix.
        encoding: list[int]
            Subword tokens of the cleaned document, featurized in the Byte-Pair space.

        Returns
        -------
        final_dict: dict[str, int]
            Language probabilities.
        """

        document_bag_of_words = numpy.bincount(encoding, minlength=config.VOCAB_SIZE)
        word_lang_map = numpy.random.randint(0, len(config.CORPUS_LANGUAGES), document_bag_of_words.sum())
        lang_dist = numpy.bincount(word_lang_map, minlength=len(config.CORPUS_LANGUAGES)) + config.GIBBS_ALPHA
        word_counter = 0
        for (sequence_number, element) in enumerate(itertools.cycle(document_bag_of_words)):
            if word_counter == config.GIBBS_SAMPLING_ITERATIONS:
                break
            if element == 0:
                continue
            else:
                for _ in range(element):
                    word_counter = word_counter + 1
                    word_lang = word_lang_map[word_counter % word_lang_map.size]
                    lang_dist[word_lang] = lang_dist[word_lang] - 1
                    lang_dist = lang_dist * frequency_history[sequence_number % config.VOCAB_SIZE]
                    sampled_lang = numpy.random.choice(len(config.CORPUS_LANGUAGES), p=lang_dist / lang_dist.sum())
                    word_lang_map[word_counter % word_lang_map.size] = sampled_lang
                    lang_dist[sampled_lang] = lang_dist[sampled_lang] + 1

        final_dist = lang_dist - config.GIBBS_ALPHA
        final_dist = final_dist / final_dist.sum()

        return final_dist

    def infer_language_distribution(self,
                                    doc_path: str) -> Dict[str, int]:
        """
        Infers document language labels by invoking the gibbs sampler sub-routine and thresholding the probabilities
        based on the configured threshold.

        Parameters
        ----------
        doc_path: str
            The file path to the document.

        Returns
        -------
        result_dict: dict[str, int]
            Language probabilities thresholded by configuration threshold.
        """

        document_encoding = copy.deepcopy(self._process_document(self.configuration, doc_path, self.parser))
        final_dist = copy.deepcopy(self._gibbs_sampler(self.configuration,
                                                       self.frequency_history,
                                                       document_encoding))
        result_dict = {}
        for (sequence_number, likelihood) in enumerate(final_dist):
            if likelihood >= self.configuration.THRESHOLD:
                result_dict[self.configuration.CORPUS_LANGUAGES[sequence_number]] = likelihood

        print(result_dict)

        return result_dict
