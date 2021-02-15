# Preprocessing utilities.
import re
from typing import Pattern


class Configuration:
    """
    Configuration store class.
    """

    # Threshold of evidence.
    THRESHOLD = 0.3

    # Gibbs sampling parameters.
    GIBBS_SAMPLING_ITERATIONS = 10000
    GIBBS_ALPHA = 25
    GIBBS_BETA = 1

    # Corpus details.
    CORPUS_DIRECTORY = "/Users/indranep/Downloads/MultiLang/"
    CORPUS_MAP_PATH = "/Users/indranep/Downloads/MultiLang/map-multilang.csv"
    CORPUS_LANGUAGES = ["English", "French", "Spanish"]

    # Tokenizer utilities.
    TOKENIZER_PATH = "/Users/indranep/Downloads/MultiLang/tokenizer-multilang.json"
    VOCAB_SIZE = 50000
    MINIMUM_FREQUENCY = 5

    # Preprocessing regexes.
    RE_URL: Pattern = re.compile(
        r"(?:^|(?<![\w/.]))"
        # protocol identifier
        # r"(?:(?:https?|ftp)://)"  <-- alt?
        r"(?:(?:https?://|ftp://|www\d{0,3}\.))"
        # user:pass authentication
        r"(?:\S+(?::\S*)?@)?"
        r"(?:"
        # IP address exclusion
        # private & local networks
        r"(?!(?:10|127)(?:\.\d{1,3}){3})"
        r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
        r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
        # IP address dotted notation octets
        # excludes loopback network 0.0.0.0
        # excludes reserved space >= 224.0.0.0
        # excludes network & broadcast addresses
        # (first & last IP address of each class)
        r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
        r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
        r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
        r"|"
        # host name
        r"(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
        # domain name
        r"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
        # TLD identifier
        r"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
        r")"
        # port number
        r"(?::\d{2,5})?"
        # resource path
        r"(?:/\S*)?"
        r"(?:$|(?![\w?!+&/]))",
        flags=re.UNICODE | re.IGNORECASE)

    RE_SHORT_URL: Pattern = re.compile(
        r"(?:^|(?<![\w/.]))"
        # optional scheme
        r"(?:(?:https?://)?)"
        # domain
        r"(?:\w-?)*?\w+(?:\.[a-z]{2,12}){1,3}"
        r"/"
        # hash
        r"[^\s.,?!'\"|+]{2,12}"
        r"(?:$|(?![\w?!+&/]))",
        flags=re.UNICODE | re.IGNORECASE)

    RE_EMAIL: Pattern = re.compile(
        r"(?:mailto:)?"
        r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}"
        r"(?:$|(?=\b))",
        flags=re.UNICODE | re.IGNORECASE)

    RE_LINEBREAK: Pattern = re.compile(r"(\r\n|[\n\v])+")
    RE_NONBREAKING_SPACE: Pattern = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)
    RE_ZWSP: Pattern = re.compile(r"[\u200B\u2060\uFEFF]+")
