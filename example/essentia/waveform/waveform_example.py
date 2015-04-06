__author__ = 'jeffreyquinn'
import essentia
import essentia.standard

from numpy import arange
from pylab import plot, savefig

def main():
    audio = essentia.standard.MonoLoader(filename="input/Bowed-Bass-C2.wav", sampleRate=2000)()
    x = arange(len(audio))
    y = audio
    plot(x, y)
    savefig("output/waveform")

if __name__ == '__main__':
    main()
