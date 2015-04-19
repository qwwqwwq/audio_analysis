__author__ = 'jeffreyquinn'
import pylab


def plot_lines(args, labels, out_fn):
    """
    Create an overlayed, multicolored plot of multiple sets of cartesian points

    :param args: list-like of 2-tuples of X and Y points to plot
    """
    lines = []
    for (idx, (xy, label)) in enumerate(zip(args, labels)):
        lines.append(pylab.plot(xy[0], xy[1], label=label))

    pylab.legend(loc=2,
                 ncol=1, mode="expand", borderaxespad=0.)
    pylab.savefig(out_fn)
    pylab.close()
