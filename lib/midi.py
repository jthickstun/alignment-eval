import numpy as np
import mido

def load_midi(filename):
    """
        input: a midi file given by path 'filename'

        returns symbolic midi data as a list of notes
        each note note is a tuple consisting of:

            pitch - integer in [0,127]
            onset - real-valued time in seconds
            offset - real-valued time in seconds

        the list of notes is sorted by onset time
    """

    midi = mido.MidiFile(filename)

    notes = []
    time = 0
    for message in midi:
        time += message.time
        # velocity == 0 equivalent to note_off, see here:
        # http://www.kvraudio.com/forum/viewtopic.php?p=4167096
        if message.type == 'note_on' and message.velocity != 0:
            # some midis seem to have timing info on channel 0
            # but not intended to be played? (e.g. ravel)
            #if message.channel==0:
            #    continue
            notes.append((message.note, time, -1))
        elif (message.type == 'note_off') or (message.type == 'note_on' and message.velocity == 0):
            # Find the last time this note was played and update that
            # entry with offset.
            for i, e in reversed(list(enumerate(notes))):
                (note, onset, offset) = e
                if note == message.note:
                    notes[i] = (note, onset, time)
                    break

    # only keep the entries with have an offset
    notes = [x for x in notes if not x[2] == -1]
    
    # sanity checks
    for note, onset, offset in notes: assert onset <= offset
    assert time == midi.length

    return notes, midi.ticks_per_beat


def load_midi_events(filename, merge=True, strip_ends=True):
    """
        input: a midi file given by path 'filename'
               optional 'merge' to merge adjacent equal run-lengths
               optional 'strip_ends' to strip empty space from beginning/end
 
        returns:
          (1) a sequence of "events" x in R^{T x 129}
              where T is the number of distinct temporal run-lengths in the midi
              a slice of the event matrix x[i] is a vector in R^{129} where

                x[0],dots,x[127] in {0,1}   (pitch indicator)
                x[128] in R^+               (duration)

              the first 128 elements are binary indicators of whether a pitch occurs in the run
              the final element is a positive real time duration that encodes the length of the run

          (2) the time of the first onset in the performance (t=0)
          (3) the time of the last onset in the performance (t=T)
    """
    midi = mido.MidiFile(filename)

    events = []

    time = 0
    cur_event = np.zeros(129)
    last_onset = 0
    for message in midi:
        time += message.time
        if message.time != 0:
            if merge and len(events) > 0 and np.array_equal(cur_event[:128], events[-1][:128]):
                events[-1][-1] += message.time
            else:
                cur_event[-1] = message.time
                events.append(cur_event)
                cur_event = np.copy(cur_event)
                cur_event[-1] = 0 

        if message.type == 'note_on' and message.velocity != 0:
            cur_event[message.note] = 1
            last_onset = time
        elif (message.type == 'note_off') or (message.type == 'note_on' and message.velocity == 0):
            cur_event[message.note] = 0

    first_onset = 0
    if not np.any(events[0][:128]):
        first_onset = events[0][-1]
        if strip_ends: events = events[1:]

    if not np.any(events[-1][:128]) and strip_ends:
        events = events[:-1]

    return np.stack(events),first_onset,last_onset


def write_midi(filename, notes, tpb):
    mid = mido.MidiFile(ticks_per_beat=tpb) # copy ticks_per_beat from source to avoid rounding errors
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    track.append(mido.MetaMessage('time_signature'))
    track.append(mido.Message('program_change', program=0))

    events = [(n[0], n[1], 'note_on') for n in notes]
    events.extend([(n[0], n[2], 'note_off') for n in notes])
    events = sorted(events, key = lambda n : n[1])

    time = t0 = 0
    for pitch,t1,eventtype in events:
        time += t1 - t0
        dt = mido.second2tick(t1 - t0,tpb,tempo)
        message = mido.Message(eventtype, note=pitch, velocity=64, time=round(dt))
        track.append(message)
        t0 = t1

    mid.save(filename)


def split(notes):
    """ heuristically find the split between the prelude and fugue in a midi file by finding a large gap """
    distinct_onsets = sorted(list(set([n[1] for n in list(notes)])))
    max_diff = max_index = 0
    for i in range(len(distinct_onsets)-1):
        diff = distinct_onsets[i+1] - distinct_onsets[i]
        if diff > max_diff:
            max_diff = diff
            max_index = i

    splitpoint = (distinct_onsets[max_index] + distinct_onsets[max_index+1])/2
    if (splitpoint < 10) or (splitpoint > distinct_onsets[-1] - 10):
        raise ValueError # sanity check failed

    return splitpoint

