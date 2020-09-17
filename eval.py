import os, sys
import numpy as np

import lib.util as util
import lib.midi as midi

def match_onsets(score_notes, perf_notes, gt_alignment, thres=.100):
    """ Onset matching heuristic """

    matched_onsets = []
    pitches,score_onsets,score_offsets = zip(*score_notes)
    gt_onsets = np.interp(score_onsets,*zip(*gt_alignment))
    for pitch,gt_onset,score_onset in zip(pitches,gt_onsets,score_onsets):
        best_dist = 1 + thres
        for perf_pitch, perf_onset, _ in perf_notes:
            if np.abs(perf_onset - gt_onset) > thres:
                 continue # not a match (too far away)
            if perf_pitch != pitch:
                 continue # not a match (wrong pitch)

            dist = np.abs(perf_onset - gt_onset)
            if dist < thres and dist < best_dist: # found a possible match
                best_match = perf_onset
                best_dist = dist

        if best_dist < thres: # found a match
            matched_onsets.append((score_onset,best_match))
            
    return matched_onsets

def evaluate(candidatedir, gtdir, scoredir, perfdir):
    mad, old_mad, rmse, old_rmse, missedpct = [[] for _ in range(5)]
    print("Performance\tTimeErr\tTimeDev\tNoteErr\tNoteDev\t%Match")
    for file in sorted([f[:-len('.midi')] for f in os.listdir(perfdir) if f.endswith('.midi')]):
        gt_alignment = np.loadtxt(os.path.join(gtdir, file + '.txt'))
        ch_alignment = np.loadtxt(os.path.join(candidatedir, file + '.txt'))
    
        score_file = os.path.join(scoredir, util.map_score(file) + '.midi')
        _,score_start,score_end = midi.load_midi_events(score_file, strip_ends=True)
    
        # truncate to the range [score_start,score_end)
        idx0 = np.argmin(score_start > gt_alignment[:,0])
        idxS = np.argmin(score_end > gt_alignment[:,0])
        gt_alignment = gt_alignment[idx0:idxS]
    
        #
        # compute our metrics
        #
    
        # linearized timings
        linearized = np.interp(gt_alignment[:,0],*zip(*ch_alignment))
        ch_alignment = gt_alignment.copy()
        ch_alignment[:,1] = linearized
    
        S = gt_alignment[-1,0] - gt_alignment[0,0]
        ds = gt_alignment[1:,0] - gt_alignment[:-1,0]
        error = list(gt_alignment[0:,1] - ch_alignment[0:,1])

        dev = [(1/2)*(e1+e2) if samesign
            else (1/2)*(e1**2+e2**2)/(e1+e2)
            for (e1,e2,samesign) in zip(np.abs(error[:-1]),np.abs(error[1:]),np.sign(error[:-1])==np.sign(error[1:]))]

        se = [(1/3)*(e1**2+e1*e2+e2**2) for (e1,e2) in zip(error[:-1],error[1:])]

        mad.append((1./S)*np.dot(dev,ds))
        rmse.append(np.sqrt((1./S)*np.dot(se,ds)))
    
        #
        # compute old metrics
        #
    
        score_notes,_ = midi.load_midi(os.path.join(scoredir, util.map_score(file) + '.midi'))
        perf_notes,_ = midi.load_midi(os.path.join(perfdir,file + '.midi'))
        matched_onsets = match_onsets(score_notes, perf_notes, gt_alignment)
        missedpct.append(100*len(matched_onsets)/len(score_notes))

        onsets, gt_onsets = zip(*matched_onsets)
        ch_aligned_onsets = np.interp(onsets,*zip(*ch_alignment))
        dev = gt_onsets - ch_aligned_onsets

        old_mad.append((1./len(dev))*np.sum(np.abs(dev)))
        old_rmse.append(np.sqrt((1./len(dev))*np.sum(np.power(dev,2))))

        print('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(file, mad[-1], rmse[-1], old_mad[-1], old_rmse[-1], missedpct[-1]))

    print('=' * 100)
    print('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format('bottomline', np.mean(mad), np.mean(rmse), np.mean(old_mad), np.mean(old_rmse), np.mean(missedpct)))

if __name__ == "__main__":
    algo = sys.argv[1]
    scoredir = sys.argv[2]
    perfdir = sys.argv[3]

    candidatedir = os.path.join('align',algo)
    gtdir = os.path.join('align','ground')
    evaluate(candidatedir, gtdir, scoredir, perfdir)

