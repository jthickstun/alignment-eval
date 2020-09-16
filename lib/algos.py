import sys
import numpy as np
import librosa, pretty_midi
import lib.midi as midi
import lib.util as util

import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={"include_dirs":np.get_include()})
import lib.gtalign as gtalign

def align_ground_truth(score_midi, perf, fs=44100, stride=512, lmbda=1.0):
    score_events,score_start,score_end = midi.load_midi_events(score_midi)
    perf_events,perf_start,perf_end = midi.load_midi_events(perf + '.midi')

    score_rep = score_events.astype(np.float32)
    perf_rep = util.pianoroll(perf_events).astype(np.float32)

    ds = stride/fs
    L = gtalign.align(score_rep,perf_rep,ds,lmbda)
    path,_ = gtalign.traceback(score_rep,perf_rep,L,ds,lmbda)

    index_alignment = [dict(path)[i] for i in range(len(score_events))]
    score_timing = score_start + np.cumsum(score_events[:,-1])
    perf_timing = [perf_start + k*(stride/fs) for k in index_alignment]
    alignment = np.array(list(zip(score_timing,perf_timing)))
    return np.insert(alignment, 0, (score_start,perf_start), axis=0)

def align_chroma(score_midi, perf, fs=44100, stride=512, n_fft=4096):
    score_synth = pretty_midi.PrettyMIDI(score_midi).fluidsynth(fs=fs)
    perf,_ = librosa.load(perf + '.wav', sr=fs)
    score_chroma = librosa.feature.chroma_stft(y=score_synth, sr=fs, tuning=0, norm=2,
                                               hop_length=stride, n_fft=n_fft)
    score_logch = librosa.power_to_db(score_chroma, ref=score_chroma.max())
    perf_chroma = librosa.feature.chroma_stft(y=perf, sr=fs, tuning=0, norm=2,
                                              hop_length=stride, n_fft=n_fft)
    perf_logch = librosa.power_to_db(perf_chroma, ref=perf_chroma.max())
    D, wp = librosa.sequence.dtw(X=score_logch, Y=perf_logch)
    path = np.array(list(reversed(np.asarray(wp))))

    return np.array([(s,t) for s,t in dict(reversed(wp)).items()])*(stride/fs)

def align_spectra(score_midi, perf, fs=44100, stride=512, n_fft=4096):
    score_synth = pretty_midi.PrettyMIDI(score_midi).fluidsynth(fs=fs)
    perf,_ = librosa.load(perf + '.wav', sr=fs)
    score_spec = np.abs(librosa.stft(y=score_synth, hop_length=stride, n_fft=n_fft))**2
    score_logspec = librosa.power_to_db(score_spec, ref=score_spec.max())
    perf_spec = np.abs(librosa.stft(y=perf, hop_length=stride, n_fft=n_fft))**2
    perf_logspec = librosa.power_to_db(perf_spec, ref=perf_spec.max())
    D, wp = librosa.sequence.dtw(X=score_logspec, Y=perf_logspec)
    path = np.array(list(reversed(np.asarray(wp))))

    return np.array([(s,t) for s,t in dict(reversed(wp)).items()])*(stride/fs)

def align_prettymidi(score_midi, perf, fs=22050, hop=512, note_start=36, n_notes=48, penalty=None):
    '''
    Align a MIDI object in-place to some audio data.
    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing some MIDI content
    audio_data : np.ndarray
        Samples of some audio data
    fs : int
        audio_data's sampling rate, and the sampling rate to use when
        synthesizing MIDI
    hop : int
        Hop length for CQT
    note_start : int
        Lowest MIDI note number for CQT
    n_notes : int
        Number of notes to include in the CQT
    penalty : float
        DTW non-diagonal move penalty
    '''
    def extract_cqt(audio_data, fs, hop, note_start, n_notes):
        '''
        Compute a log-magnitude L2-normalized constant-Q-gram of some audio data.
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data to compute CQT of
        fs : int
            Sampling rate of audio
        hop : int
            Hop length for CQT
        note_start : int
            Lowest MIDI note number for CQT
        n_notes : int
            Number of notes to include in the CQT
        Returns
        -------
        cqt : np.ndarray
            Log-magnitude L2-normalized CQT of the supplied audio data.
        frame_times : np.ndarray
            Times, in seconds, of each frame in the CQT
        '''
        # Compute CQT
        cqt = librosa.cqt(
            audio_data, sr=fs, hop_length=hop,
            fmin=librosa.midi_to_hz(note_start), n_bins=n_notes)
        # Transpose so that rows are spectra
        cqt = cqt.T
        # Compute log-amplitude
        cqt = librosa.amplitude_to_db(librosa.magphase(cqt)[0], ref=cqt.max())
        # L2 normalize the columns
        cqt = librosa.util.normalize(cqt, norm=2., axis=1)
        # Compute the time of each frame
        times = librosa.frames_to_time(np.arange(cqt.shape[0]), fs, hop)
        return cqt, times

    audio_data, _ = librosa.load(perf + '.wav', fs)
    midi_object = pretty_midi.PrettyMIDI(score_midi)
    # Get synthesized MIDI audio
    midi_audio = midi_object.fluidsynth(fs=fs)
    # Compute CQ-grams for MIDI and audio
    midi_gram, midi_times = extract_cqt(
        midi_audio, fs, hop, note_start, n_notes)
    audio_gram, audio_times = extract_cqt(
        audio_data, fs, hop, note_start, n_notes)
    # Compute distance matrix; because the columns of the CQ-grams are
    # L2-normalized we can compute a cosine distance matrix via a dot product
    distance_matrix = 1 - np.dot(midi_gram, audio_gram.T)
    D, wp = librosa.sequence.dtw(C=distance_matrix)
    path = np.array([(s,t) for s,t in dict(reversed(wp)).items()])
    result = [(midi_times[x[0]], audio_times[x[1]]) for x in path]
    return np.array(result)

