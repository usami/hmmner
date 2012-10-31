#!/usr/bin/env python

import re

def symbolize(word, options):
    if options.digits and word.isdigit():
        return '_DIGITS_'
    elif options.allc and word.isupper(): # all uppercase
        return '_ALL_CAPITAL_'
    elif options.initc and word.istitle(): # first word is capitalized
        return '_INIT_CAPITAL_'
    else:
        return '_RARE_'
