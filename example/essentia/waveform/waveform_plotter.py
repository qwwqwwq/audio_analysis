__author__ = 'jeffreyquinn'
import argparse

import essentia
import essentia.standard
import numpy
import pandas
from numpy import arange, linspace, max
from pylab import plot, savefig, axvline, close
from audio.algorithms import find_peaks, spectrum, get_metadata, get_duration, resample_audio, moving_average, \
    get_spline_function, get_gradient, energy_buckets, moving_max, load_partial_audio


PLOT_SAMPLE_RATE = 1000


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


def write_raw(arr, dest):
    pandas.DataFrame(arr).to_csv(dest)


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