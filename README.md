Hidden Markov Model Named Entity Tagger
======

Train a tagger on a training data and predict tags for input words by 
using HMM and Viterbi decoding.


Usage: python hmm_ne_tagger.py [options] [command] [counts_file] < [input_file]

Command:
        trigram     prints the log probabilities for the input_file
        decode      decodes the tag sequences for the input_file with the log probabilities

Options:
        -h, --help          show this help message and exit
        -d, --digits        use _DIGITS_ symbol for the words which are compounded
                            only digits
        -i, --init-capital  use _INIT_CAPITAL_ symbol for the words start with
                            uppercase character
        -a, --all-capital   use _ALL_CAPITAL_ symbol for the all capitalized words
