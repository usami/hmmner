#!/usr/bin/env python

__author__ = 'Yu Usami <yu2118@columbia.edu>'
__date__ = '$Sep 16, 2012'

import sys
from collections import defaultdict
from math import log
from optparse import OptionParser

from symbolize import symbolize

"""
Hidden Markov Model Named Entity Tagger
Train a tagger with a data file and predict tags for input words with
HMM and Viterbi decoding.
"""


def read_counts(counts_file):
    """
    Read lines from a file which includes frequency counts in a training 
    corpus, return an iterator yields each entity as a list. 
    """
    try:
        fi = open(counts_file, 'r')
    except IOError:
        sys.stderr.write('ERROR: Cannot open %s.\n' % counts_file)
        sys.exit(1)

    for line in fi:
        fields = line.strip().split(' ')
        yield fields # yields a list of fields


class HMMNETagger():
    """
    Stores emission counts and n-gram (n = 1,2,3) counts.
    Estimates maximum likelihood transition parameters from bigram and
    trigram counts.
    Reads words from stdin, and decodes each sentence to the tag sequence.
    Predicts tags with the Viterbi algorithm.
    """
    def __init__(self, options):
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for _ in xrange(3)] # [1-gram, 2-gram, 3-gram]
        self.symbolize_option = options 

    def train(self, counts_file):
        """
        Stores emission counts and ngram counts from a file which contains
        frequencies.
        """
        for l in read_counts(counts_file):
            n, count_type, args = int(l[0]), l[1], l[2:]
            if count_type == 'WORDTAG': # emission counts
                self.emission_counts[tuple(args)] = n
            else: # ngram counts
                self.ngram_counts[len(args) - 1][tuple(args)] = n

    def q(self, y3, y1, y2):
        """
        Return a maximum likelihood transition parameter for a given
        trigram y1 y2 y3.
        """
        return float(self.ngram_counts[2][tuple([y1, y2, y3])]) / self.ngram_counts[1][tuple([y1, y2])]

    def e(self, x, y):
        """
        Return a maximum likelihood emission parameter for a given word 
        and tag pair x y.
        """
        return float(self.emission_counts[tuple([y, x])]) / self.ngram_counts[0][tuple([y])]

    def trigram(self, input):
        """
        Reads state trigrams form input, and prints the log probability for 
        each trigram.
        """
        for l in input:
            line = l.strip()
            y1, y2, y3 = line.split(' ')
            if line:
                print line, log(self.q(y3, y1, y2))

    def decode(self, input):
        """
        Reads words from input, and decodes each sentence to the tag 
        sequence.
        Predicts tag sequences with the Viterbi algorithm.
        Writes words with corresponding tags and log probabilities to 
        stdout.
        """
        S = [s[0] for s in self.ngram_counts[0].keys()] # Set of tags
        _S = [s[0] for s in self.ngram_counts[0].keys()]
        _S.append('*') # _S includes '*' tag
        X = ['*'] # X stores each sentence. X[0] = '*', X[i] = xi
        for l in input:
            x = l.strip()
            if x: # Word
                X.append(x)
            else: # End of a sentence
                n = len(X) - 1 # the length of the sentence
                pi = defaultdict(float) # DP table PI
                bp = {} # back pointer

                # Initialize DP table
                for u in _S:
                    for v in _S:
                        pi[tuple([0, u, v])] = 0
                pi[tuple([0, '*', '*'])] = 1

                # Viterbi algorithm
                for k in xrange(1, n + 1):
                    for u in _S:
                        for v in S: # v will not be '*'  
                            max_score = 0
                            tag = None
                            for w in _S:
                                if sum([self.emission_counts[tuple([y, X[k]])] for y in S]) < 5: # If the word X[k] is rare word or unseen word in the training corpus,
                                    x = symbolize(X[k], self.symbolize_option) # use RARE word probability
                                else:
                                    x = X[k]
                                try:
                                    score = pi[tuple([k-1, w, u])] * self.q(v, w, u) * self.e(x, v)
                                    if max_score < score:
                                        max_score = score
                                        tag = w
                                except:
                                    pass
                            pi[tuple([k, u, v])] = max_score # Update DP table entry
                            bp[tuple([k, u, v])] = tag

                # Find tag sequence
                Y = ['*'] # Y stores tag sequence for X. Y[0] = '*', Y[i] = yi
                Y.extend(n * [None])
                max_score = None
                tag = None
                for u in _S:
                    for v in _S:
                        if self.ngram_counts[1][tuple([u, v])]:
                            score = pi[tuple([n, u, v])] * self.q('STOP', u, v)
                            if max_score is None or max_score < score:
                                max_score = score
                                tag = [u, v]
                Y[n-1] = tag[0]
                Y[n] = tag[1]
                for k in xrange(n - 2, 0, -1):
                    Y[k] = bp[tuple([k + 2, Y[k + 1], Y[k + 2]])]

                # Write result
                prev = '*'
                for k in xrange(1, n + 1):
                    print X[k], Y[k], log(pi[tuple([k, prev, Y[k]])])
                    prev = Y[k]
                print ''

                X = ['*'] # set for the next sentence

if __name__ == '__main__':
    opt_parser = OptionParser(usage="""Usage: python hmm_ne_tagger.py [options] [command] [counts_file] < [input_file]

Command:
        trigram     prints the log probabilities for the input_file
        decode      decodes the tag sequences for the input_file with the log probabilities""")

    opt_parser.add_option(
            '-d', '--digits',
            dest='digits',
            action='store_true', default=False,
            help='use _DIGITS_ symbol for the words which are compounded only digits'
            )
    opt_parser.add_option(
            '-i', '--init-capital',
            dest='initc',
            action='store_true', default=False,
            help='use _INIT_CAPITAL_ symbol for the words start with uppercase character'
            )
    opt_parser.add_option(
            '-a', '--all-capital',
            dest='allc',
            action='store_true', default=False,
            help='use _ALL_CAPITAL_ symbol for the all capitalized words'
            )

    options, args = opt_parser.parse_args() # parse options

    if len(args) != 2: # Expect exactly two arguments.
        opt_parser.print_help()
        sys.exit(2)

    tagger = HMMNETagger(options) # Initialize a named entity tagger
    tagger.train(args[1]) # Train with the counts
    if args[0] == 'trigram': # Command: trigram
        tagger.trigram(sys.stdin) 
    elif args[0] == 'decode': # Command: decode
        tagger.decode(sys.stdin)
    else: # Command Error
        print >> sys.stderr, 'ERROR: Use trigram or decode as command.\n'
        opt_parser.print_help()
        sys.exit(2)
