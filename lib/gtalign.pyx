import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef align(float[:,:] score, float[:,:] perf, float ds, float lmbda):
    cdef float[:] score_timing = score[:,128]
    cdef float prior = (ds*perf.shape[0])/np.cumsum(score_timing)[len(score_timing)-1] # slope = rise/run
    
    cdef np.ndarray[np.float32_t, ndim=2] npL = np.full((len(score),len(perf)), np.inf, dtype=np.float32)
    cdef float[:,:] L = npL # memory view for cheap access

    cdef float cost,tmp,sj,instantaneous_tempo,incremental_cost,R
    cdef int j,k,m,i

    cdef np.ndarray[np.float32_t, ndim=2] npC = np.empty((len(score),len(perf)), dtype=np.float32)
    cdef float[:,:] local_cost = npC

    # precompute the local cost of aligning score[j] with perf[k]
    for j in range(0,len(score)):
        for k in range(0,len(perf)):
            local_cost[j,k] = 0
            for i in range(128):
                tmp = score[j,i] - perf[k,i]
                local_cost[j,k] += tmp if tmp > 0 else -tmp

    # base case j = 0
    sj = score_timing[0]
    incremental_cost = 0
    for k in range(0,len(perf)):
        instantaneous_tempo = (k*ds)/sj
        tmp = instantaneous_tempo - prior
        R = lmbda*tmp*tmp
        incremental_cost += local_cost[0,k]
        L[0,k] = incremental_cost*ds + R

    L[0,0] = 0 # base case
    # k > 0 is unreachable when j = 0 (no up moves allowed) so leave L[0,k] = np.inf for k > 0
    for j in range(1,len(score)): # x-axis (score)
        sj = score_timing[j]
        for k in range(0,len(perf)): # y-axis (performance)
            incremental_cost = 0
            # go backward to incrementally compute mean cost

            # limit tempo search to [.5*prior,2*prior)
            #minm = max(0,k - int(2*prior*sj/ds))
            #maxm = min(k+1,k - int(.5*prior*sj/ds))
            #for m in reversed(range(minm,maxm)): # unrestricted range would be range(0,k+1)
            for m in reversed(range(0,k+1)):
                # instantaneous tempo over score event j is
                # elapsed duration in the performance (k-m)*ds divided by elapsed time in the score score_timing[j]
                instantaneous_tempo = ((k-m)*ds)/sj
                #if instantaneous_tempo < .5*prior or instantaneous_tempo > 2*prior: continue

                # I *think* the python power operator translates to the same as repeat multiply here
                # but in the non-cython setting (i.e. traceback it casts to 64-bit) so we need to
                # do the multiply manually in numpy 32-bit floats over there; doing the same here
                # seems safest...
                tmp = instantaneous_tempo - prior
                R = lmbda*tmp*tmp

                # cost is cost of advancing from previous state L[j-1,m]
                # plus the cost of aligning score[j] to (performance[m],performance[k]]
                # plus regularization for the tempo required to make this step
                cost = L[j-1,m] + incremental_cost*ds + R

                # track the minimum cost to reach state L[j,k]
                if cost < L[j,k]: L[j,k] = cost

                # the cost of aligning score[j] to (performance[m],...,performance[k]]
                #     =  cost of aligning score[j] to (performance[m+1],...,performance[k]]
                #     + cost of aligning score[j] 
                #     = int_{m}^k \|score[j] - perf[m]\|_1 \,dt
                incremental_cost += local_cost[j,m]

    return npL

def traceback(score, perf, L, ds, lmbda):
    # 32-bit arithmetic so that floating-point equalities work out
    ds = np.float32(ds)
    lmbda = np.float32(lmbda)

    score_timing = score[:,128]
    prior = (ds*np.float32(len(perf)))/np.cumsum(score_timing)[len(score_timing)-1]

    A,C = [],[]
    k = len(perf)-1
    for j in reversed(range(1,len(score))):
        sj = score_timing[j]
        incremental_cost = np.float32(0)
        for m in reversed(range(0,k+1)):
            instantaneous_tempo = ((np.float32(k)-np.float32(m))*ds)/sj
            #if instantaneous_tempo < .5*prior or instantaneous_tempo > 2*prior: continue

            tmp = instantaneous_tempo - prior
            R = lmbda*tmp*tmp

            if L[j,k] == L[j-1,m] + incremental_cost*ds + R:
                A.append((j,k))
                C.append(L[j,k])
                k = m
                break      # found the match

            for i in range(128):
                tmp = score[j,i] - perf[m,i]
                incremental_cost += tmp if tmp > 0 else -tmp

        else: assert False # we had to come from somewhere...

    A.append((0,k))
    C.append(L[0,k])

    return list(reversed(A)),list(reversed(C))
