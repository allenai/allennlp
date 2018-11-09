#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True

cimport cython
cimport numpy as np
np.import_array()

from libc.math cimport sqrt, log, exp
from libc.stdlib cimport rand, RAND_MAX
from libcpp.vector cimport vector
from libc.stdint cimport int64_t, int8_t


import numpy as np


# type definitions
#cdef extern from "stdint.h":
#    ctypedef unsigned long long uint32_t

cdef type DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef int64_t INT_t

cdef inline INT_t int_min(INT_t a, INT_t b) nogil: return a if a <= b else b
cdef inline INT_t int_max(INT_t a, INT_t b) nogil: return a if a >= b else b


@cython.boundscheck(False)
cdef INT_t fast_choice(INT_t n_words, INT_t n_samples,
                       INT_t* samples) nogil:

    cdef INT_t n_tries = 0
    cdef INT_t n_chosen = 0
    cdef DTYPE_t log_n_words_p1 = log(n_words + 1.0)

    cdef INT_t sample_id

    cdef vector[int8_t] chosen_samples
    chosen_samples.reserve(n_words)
    cdef size_t k
    for k in range(n_words):
        chosen_samples[k] = 0

    cdef DTYPE_t rand_float

    while n_chosen < n_samples:
        n_tries += 1

        # choose a sample
        rand_float = (<DTYPE_t> rand()) / (<DTYPE_t> RAND_MAX)
        sample_id = <INT_t>exp(rand_float * log_n_words_p1)
        sample_id -= 1
        sample_id = int_max(int_min(sample_id, n_words - 1), 0)

        # check to see if it's already chosen
        if chosen_samples[sample_id] == 0:
            # haven't chosen this yet
            chosen_samples[sample_id] = 1
            samples[n_chosen] = sample_id
            n_chosen += 1

    return n_tries

def choice(n_words, n_samples):
    '''calls np.random.choice(
            n_words, n_samples, replace=False, p=self._probs),
        computing the probability on the fly.
    Returns (samples, num_tries)
    '''
    #torch.Tensor(n_samples, dtype='int64')
    cdef np.ndarray[INT_t, ndim=1, mode='c'] samples

    # pointers to data arrays
    cdef INT_t* samples_ptr

    samples = np.empty(n_samples, dtype=np.int64)

    n_tries = fast_choice(n_words, n_samples, <INT_t*> samples.data)
    return samples, n_tries
