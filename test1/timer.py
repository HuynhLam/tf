'''
Huynh Lam, Oulu 2017

Using Timer() to measure eslaped time for a specific code block.
Usage:
    with Timer() as tm:
        pass
    print("=> eslaped time: {0:.2f} sec".format(tm.secs))
'''

import time

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs
