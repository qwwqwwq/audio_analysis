__author__ = 'jeffreyquinn'
import essentia
import essentia.standard
import numpy
import pandas

from .util import Metadata
from numpy import linspace
from scipy.interpolate import spline


def spline_audio(xold, audio, num_points=300):
    xnew = linspace(xold.min(), xold.max(), num_points)
    smooth = spline(audio, xold, xnew)
    return xnew, smooth


def find_key(audio):
    key, scale, strength = essentia.standard.KeyExtractor()(audio)
    return key


def find_peaks(audio, min_percentile=80):
    lower_bound = numpy.percentile(audio, min_percentile)
    audio = numpy.nan_to_num(audio)
    peak_positions, peak_amplitudes = \
        essentia.standard.PeakDetection()(audio)
    print "Found %d peaks" % len(peak_positions)
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


def get_gradient(audio):
    return numpy.gradient(audio)


def energy_buckets(audio, audio_length_seconds, buckets_per_second=20):
    n_buckets = audio_length_seconds * buckets_per_second
    arrs = numpy.array_split(audio, n_buckets)
    return numpy.arange(0, audio_length_seconds, audio_length_seconds / float(n_buckets)), \
           numpy.array([essentia.standard.Energy()(sub_arr) for sub_arr in arrs])


def moving_max(audio, window_size=6):
    return pandas.rolling_max(audio, 6)


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