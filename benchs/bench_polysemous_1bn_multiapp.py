# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

import os
import sys
import time
import numpy as np
import re
import faiss
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing


#################################################################
# I/O functions
#################################################################


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

# we mem-map the biggest files to avoid having them in memory all at
# once


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


#################################################################
# Bookkeeping
#################################################################


napps     = sys.argv[1]
dbname    = []
index_key = []

for i in range(int(napps)):
    dbname.append(sys.argv[2+i])

for i in range(int(napps)):
    index_key.append(sys.argv[2+int(napps)+i])


parametersets = sys.argv[2+2*int(napps):]

tmpdir = '/tmp/bench_polysemous'

if not os.path.isdir(tmpdir):
    print "%s does not exist, creating it" % tmpdir
    os.mkdir(tmpdir)


#################################################################
# Prepare dataset
#################################################################

if dbname[0].startswith('SIFT'):
    xb = mmap_bvecs('bigann/bigann_base.bvecs')
    xq = mmap_bvecs('bigann/bigann_query.bvecs')
    xt = mmap_bvecs('bigann/bigann_learn.bvecs')
elif dbname[0] == 'Deep1B':
    xb = mmap_fvecs('deep1b/base.fvecs')
    xq = mmap_fvecs('deep1b/deep1B_queries.fvecs')
    xt = mmap_fvecs('deep1b/learn.fvecs')
    # deep1B's train is is outrageously big
    xt = xt[:10 * 1000 * 1000]
    gt = ivecs_read('deep1b/deep1B_groundtruth.ivecs')
else:
    print >> sys.stderr, 'unknown dataset', dbname
    sys.exit(1)


nq, d = xq.shape

xbmulti = []
gtmulti = []
nbmulti = []
for i in range(int(napps)):
    print "Preparing dataset", dbname[i]

    if dbname[i].startswith('SIFT'):
        # SIFT1M to SIFT1000M
        dbsize = int(dbname[i][4:-1])
        # trim xb to correct size
        xbmulti.append(xb[:dbsize * 1000 * 1000])
        #print("Loading:  bigann/gnd/idx_%dM.ivecs" % dbsize)
        gtmulti.append(ivecs_read('bigann/gnd/idx_%dM.ivecs' % dbsize))

    elif dbname[i] == 'Deep1B':
        print "Multiapp is not supported"
        sys.exit(1)
    else:
        print >> sys.stderr, 'unknown dataset', dbname[i]
        sys.exit(1)

    print "sizes: B %s Q %s T %s gt %s" % (xbmulti[i].shape, xq.shape, xt.shape, gtmulti[i].shape)
    nb, d = xb.shape
    nbmulti.append(nb)
    assert gtmulti[i].shape[0] == nq


#################################################################
# Training
#################################################################


def choose_train_size(index_key):

    # some training vectors for PQ and the PCA
    n_train = 256 * 1000

    if "IVF" in index_key:
        matches = re.findall('IVF([0-9]+)', index_key)
        ncentroids = int(matches[0])
        n_train = max(n_train, 100 * ncentroids)
    elif "IMI" in index_key:
        matches = re.findall('IMI2x([0-9]+)', index_key)
        nbit = int(matches[0])
        n_train = max(n_train, 256 * (1 << nbit))
    return n_train


def get_trained_index(tmpdir1, dbname1, index_key1):
    filename = "%s/%s_%s_trained.index" % (
        tmpdir1, dbname1, index_key1)

    if not os.path.exists(filename):
        index = faiss.index_factory(d, index_key1)

        n_train = choose_train_size(index_key1)

        xtsub = xt[:n_train]
        print "Keeping %d train vectors" % xtsub.shape[0]
        # make sure the data is actually in RAM and in float
        xtsub = xtsub.astype('float32').copy()
        index.verbose = True

        t0 = time.time()
        index.train(xtsub)
        index.verbose = False
        print "train done in %.3f s" % (time.time() - t0)
        print "storing", filename
        faiss.write_index(index, filename)
    else:
        print "loading", filename
        index = faiss.read_index(filename)
    return index


#################################################################
# Adding vectors to dataset
#################################################################

def rate_limited_imap(f, l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


def matrix_slice_iterator(x, bs):
    " iterate over the lines of x in blocks of size bs"
    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    return rate_limited_imap(
        lambda (i0, i1): x[i0:i1].astype('float32').copy(),
        block_ranges)


def get_populated_index(tmpdir1, dbname1, index_key1, dbid):

    filename = "%s/%s_%s_populated.index" % (
        tmpdir1, dbname1, index_key1)

    if not os.path.exists(filename):
        index = get_trained_index(tmpdir1,dbname1, index_key1)
        i0 = 0
        t0 = time.time()
        for xs in matrix_slice_iterator(xbmulti[dbid], 100000):
            i1 = i0 + xs.shape[0]
            print '\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0),
            sys.stdout.flush()
            index.add(xs)
            i0 = i1
        print
        print "Add done in %.3f s" % (time.time() - t0)
        print "storing", filename
        faiss.write_index(index, filename)
    else:
        print "loading", filename
        index = faiss.read_index(filename)
    return index


#################################################################
# Perform searches
#################################################################


# make sure queries are in RAM
xq = xq.astype('float32').copy()

def perform_search(dbid):
    if parametersets == ['autotune'] or parametersets == ['autotuneMT']:

        if parametersets == ['autotune']:
            faiss.omp_set_num_threads(1)

        # setup the Criterion object: optimize for 1-R@1
        crit = faiss.OneRecallAtRCriterion(nq, 1)
        # by default, the criterion will request only 1 NN
        crit.nnn = 100
        crit.set_groundtruth(None, gtmulti[dbid].astype('int64'))

        # then we let Faiss find the optimal parameters by itself
        print "exploring operating points"

        t0 = time.time()
        op = ps.explore(index[dbid], xq, crit)
        print "Done in %.3f s, available OPs:" % (time.time() - t0)

        # opv is a C++ vector, so it cannot be accessed like a Python array
        opv = op.optimal_pts
        print "%-40s  1-R@1     time" % "Parameters"
        for i in range(opv.size()):
            opt = opv.at(i)
            print "%-40s  %.4f  %7.3f" % (opt.key, opt.perf, opt.t)

    else:

        # we do queries in a single thread
        faiss.omp_set_num_threads(1)
        if dbid ==0:
            print ' ' * len(parametersets[0]), '\t', 'dbid \t R@1 \t \t R@10 \t \t R@100 \t \t SingleQtime \t %pass'

        for param in parametersets:
           # print param, '\t',
            ps[dbid].set_index_parameters(index[dbid], param)
            t0 = time.time()
            ivfpq_stats[dbid].reset()
            D, I = index[dbid].search(xq, 100)
            t1 = time.time()
            nok = []
            for rank in 1, 10, 100:
                n_ok = (I[:, :rank] == gtmulti[dbid][:, :1]).sum()
                nok.append(n_ok / float(nq))
                #print "%.4f" % (n_ok / float(nq)),
            print "%s \t %d\t %.4f \t %.4f \t %.4f \t %8.3f \t %5.2f" % (param, dbid,nok[0], nok[1], nok[2], ((t1 - t0) * 1000.0 / nq), (ivfpq_stats[dbid].n_hamming_pass * 100.0 / ivfpq_stats[dbid].ncode))
            #print "%5.2f" % (ivfpq_stats[dbid].n_hamming_pass * 100.0 / ivfpq_stats[dbid].ncode)
            sys.stdout.flush()


start = time.time()

ps = []
ivfpq_stats = []
index = []


for dbid in range(int(napps)):
    ps.append(faiss.ParameterSpace())

    index.append(get_populated_index(tmpdir, dbname[dbid], index_key[dbid], dbid))
    ps[dbid].initialize(index[dbid])

    # a static C++ object that collects statistics about searches
    ivfpq_stats.append(faiss.cvar.indexIVFPQ_stats)

t1 = time.time()
print "Init Time = %.3f s\n" % (t1-start)



p = ThreadPool(int(napps))
p.map(perform_search, range(int(napps)))



