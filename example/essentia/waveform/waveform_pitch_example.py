__author__ = 'jeffreyquinn'
import essentia
import essentia.standard
import argparse
import numpy
import pandas

from numpy import arange, linspace, max
from pylab import plot, savefig, text, axvline, figure, close
from collections import namedtuple
from scipy.interpolate import spline

PLOT_SAMPLE_RATE = 1000

def spline_audio(xold, audio, num_points=300):
    xnew = linspace(xold.min(), xold.max(), num_points)
    smooth = spline(audio, xold, xnew)
    return xnew, smooth

def moving_max(audio, window_size=6):
    return pandas.rolling_max(audio, 6)

Metadata = namedtuple('Metadata',
                      ('title', # the title of the track
'artist', # the artist of the track
'album', # the album on hich this track appears
'comment', # the comment field stored in the tags
'genre', # the genre as stored in the tags
'track', # the track number
'year', # the year of publication
'length', # the length of the track, in seconds
'bitrate', # the bitrate of the track [kb/s]
'sampleRate', # the sample rate
'channels' # the number of channels
))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Input audio file (.wav, .mp3, etc..)')
    parser.add_argument('-s', '--start-time', dest='start_time', type=int, default=0)
    parser.add_argument('-e', '--end-time', dest='end_time', type=int, default=5)
    parser.add_argument('--amplitude', dest='amplitude', action='store_true', default=False)
    parser.add_argument('--find-peaks', dest='find_peaks', action='store_true', default=False)
    parser.add_argument('--spline', dest='spline', action='store_true', default=False)
    parser.add_argument('--spectrum', dest='spectrum', action='store_true', default=False)
    parser.add_argument('--moving-average', dest='moving_average', action='store_true', default=False)
    parser.add_argument('--moving-max', dest='moving_max', action='store_true', default=False)
    parser.add_argument('--gradient', dest='gradient', action='store_true', default=False)
    parser.add_argument('--energy-buckets', dest='energy_buckets', action='store_true', default=False)
    return parser.parse_args()


def find_key(audio):
    key, scale, strength = essentia.standard.KeyExtractor()(audio)
    return key


def find_peaks(audio, min_percentile=80):
    lower_bound = numpy.percentile(audio, min_percentile)
    audio = numpy.nan_to_num(audio)
    peak_positions, peak_amplitudes = \
        essentia.standard.PeakDetection()(audio)
    print "Found %d peaks" % len(peak_positions)
    print peak_positions
    return peak_positions


def spectrum(audio):
    spec = essentia.standard.Spectrum()(audio)
    return spec


def beat_finder(audio):
    bpm, ticks, confidence, estimates, bpmIntervals = essentia.standard.RhythmExtractor2013()(audio)
    return bpm, ticks, estimates, bpmIntervals


def get_metadata(filename):
    return Metadata(*essentia.standard.MetadataReader(filename=filename)())


def get_duration(audio, samplerate):
    '''

    :param audio: numpy.array
    :param samplerate: integer (Hz)
    :return: real (seconds)
    '''
    return essentia.standard.Duration(sampleRate=samplerate)(audio)


def resample_audio(audio, old_rate, new_rate=1000):
    return essentia.standard.Resample(inputSampleRate=old_rate, outputSampleRate=new_rate)(audio)


def rms_audio(audio):
    return essentia.standard.RMS()(audio)


def moving_average(audio, size=10):
    return essentia.standard.MovingAverage(size=size)(audio)


def get_spline_function(x, y):
    return essentia.standard.CubicSpline(xPoints=x, yPoints=y)


def write_raw(arr, dest):
    pandas.DataFrame(arr).to_csv(dest)


def get_gradient(audio):
    return numpy.gradient(audio)


def load_partial_audio(filename, start_time, end_time):
    """
    Load a section of audio from a file

    :param filename: String, path of audio file
    :param start_time: sample start time, in seconds
    :param end_time: sample end time, in seconds
    :return: essentia.array
    """
    audio = essentia.standard.MonoLoader(filename=filename)()
    metadata = get_metadata(filename)
    return audio[(metadata.sampleRate * start_time):(metadata.sampleRate * end_time)]


def energy_buckets(audio, audio_length_seconds, buckets_per_second=20):
    n_buckets = audio_length_seconds * buckets_per_second
    arrs = numpy.array_split(audio, n_buckets)
    return numpy.arange(0, audio_length_seconds, audio_length_seconds/float(n_buckets)), \
           numpy.array([essentia.standard.Energy()(sub_arr) for sub_arr in arrs])


def main():
    args = parse_args()
    metadata = get_metadata(args.filename)
    audio = load_partial_audio(args.filename, args.start_time, args.end_time)
    sample_seconds = get_duration(audio, metadata.sampleRate)
    audio = resample_audio(audio, metadata.sampleRate, PLOT_SAMPLE_RATE)

    if args.energy_buckets:
        _, audio = energy_buckets(audio, sample_seconds)
        print "Created %d energy points" % len(audio)
        audio = essentia.array(audio)
        # plot(energies_x, energies_y)
        # savefig("output/energy_buckets")
        # close()

    if args.amplitude:
        print 'Amplitude'
        audio = essentia.array(numpy.absolute(audio))

    write_raw(audio, 'output/raw_audio.csv')
    x = essentia.array(linspace(0, sample_seconds, len(audio)))

    if args.spline:
        print 'Spline'
        f = get_spline_function(x, audio)
        audio = essentia.array(numpy.vectorize(lambda x: f(x)[0])(audio))

    if args.moving_max:
        print 'Moving Max'
        audio = essentia.array(moving_max(audio, window_size=50))

    if args.moving_average:
        print 'Moving Average'
        audio = moving_average(audio, size=6)

    plot(x, audio)

    if args.find_peaks:
        # print beat ticks
        print 'Find Peaks'
        ticks = find_peaks(audio) * max(x)
        for tick in ticks:
            axvline(tick, ymin=0, ymax=0.1, color='red')

    if args.gradient:
        print 'Gradient'
        gradient = get_gradient(audio)
        write_raw(gradient, 'output/gradient.csv')
        plot(x, gradient, color="yellow")

    savefig('output/waveform')
    close()

    if args.spectrum:
        print 'Spectrum'
        spec = spectrum(audio)
        write_raw(spec, 'output/raw_spec.csv')
        plot(arange(len(spec)), spec)
        savefig('output/spectrum')
        close()


if __name__ == '__main__':
    main()