__author__ = 'jeffreyquinn'
import argparse
import essentia

from audio.algorithms import get_metadata, get_duration, resample_audio, energy_buckets, load_partial_audio
from numpy import linspace
from viz import plotting


PLOT_SAMPLE_RATE = 1000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+',
                        help='Input audio file (.wav, .mp3, etc..)')
    parser.add_argument('-o', '--output', dest='output', type=str)
    parser.add_argument('-s', '--start-time', dest='start_time', type=int, default=0)
    parser.add_argument('-e', '--end-time', dest='end_time', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    audios = []
    for filename in args.filename:
        metadata = get_metadata(filename)
        audio = load_partial_audio(filename, args.start_time, args.end_time)
        sample_seconds = get_duration(audio, metadata.sampleRate)
        audio = resample_audio(audio, metadata.sampleRate, PLOT_SAMPLE_RATE)
        _, audio = energy_buckets(audio, sample_seconds)
        x = essentia.array(linspace(0, sample_seconds, len(audio)))
        print "Created %d energy points" % len(audio)
        audio = essentia.array(audio)
        audios.append((x, audio))

    plotting.plot_lines(audios, args.filename, args.output)

if __name__ == '__main__':
    main()





