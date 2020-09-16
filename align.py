import os, sys, errno, time, multiprocessing
import numpy as np

import lib.util as util
import lib.algos as algos

def align_and_save(align, perf, perfdir, score_dir, outdir):
    perf_transcript = os.path.join(perfdir, perf)
    score = os.path.join(scoredir,util.map_score(perf) + '.midi')
    alignment = align(score, perf_transcript)
    np.savetxt(os.path.join(outdir, perf + '.txt'), alignment, fmt='%f\t', header='score\t\tperformance')

algo_functions = {
    'ground' : 'align_ground_truth',
    'spectra' : 'align_spectra',
    'chroma' : 'align_chroma',
    'cqt' : 'align_prettymidi'
}

if __name__ == "__main__":
    algo = sys.argv[1]
    scoredir = sys.argv[2]
    perfdir = sys.argv[3]
    parallel = int(sys.argv[4])

    outdir = os.path.join('align',algo)
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    start_time = time.time()
    performances = sorted([f[:-len('.midi')] for f in os.listdir(perfdir) if f.endswith('.midi')])
    alignment_algo = getattr(algos, algo_functions[algo])
    if parallel > 0:
        print('Computing {} alignments (parallel)'.format(algo))
        args = [(alignment_algo, perf, perfdir, scoredir, outdir) for perf in performances]
        with multiprocessing.Pool(parallel) as p: p.starmap(align_and_save, args)
    else:
        print('Computing {} alignments'.format(algo))
        for perf in performances:
            print('  ', perf, end=' ')
            t0 = time.time()
            perf_audio = os.path.join(perfdir, perf + '.wav')
            score = os.path.join(scoredir,util.map_score(perf) + '.midi')

            alignment = alignment_algo(score, perf_audio)
            np.savetxt(os.path.join(outdir, perf + '.txt'), alignment, fmt='%f\t', header='score\t\tperformance')

            t1 = time.time()-t0
            print('({} seconds)'.format(t1))
        total += t1

    print('Elapsed time: {} seconds'.format(time.time()-start_time))
