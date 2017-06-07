#! /usr/bin/env python3

import numpy
import csv
from scipy import optimize, signal

from matplotlib import pyplot as plt

TEST_DATA_FILE = 'Epi SiGe_1 normalized.txt'
INTENSITY_PEAK_THRESHOLD = 0.2

def fit_eqn(w, i0, q, wp, gamma):
    """The equation to fit."""
    return i0 * (q + (w - wp) / gamma) / (1 + ((w - wp) / gamma)**2)


def fit_three_curves(w, *args):

    # Split args into 3 sets of arguments
    res = 0.
    l = len(args)
    for ndx in range(0, l, 4):
        sub_args = args[ndx:min(ndx + 4, l)]
        res += fit_eqn(w, *sub_args)
    return res

def fit_n_curves(n):

    def the_func(w, *args):
        #print(args)
        l = len(args)
        assert l == n * 4

        res = 0.0

        # import ipdb; ipdb.set_trace()
        for ndx in range(0, l, 4):

            sub_args = args[ndx:min(ndx+4, l)]
            #print(sub_args)
            res += fit_eqn(w, *sub_args)

        return res

    return the_func


def build_fit_guess(wavenumber, intensity, peak_inds):
    fit_guess = []

    for pk_ind in peak_inds:
        fit_guess += [intensity[pk_ind], 1., wavenumber[pk_ind], 10.]

    return fit_guess


def read_data(data_file):
    """Read the Data from a file and create typed numpy arrays."""
    with open(data_file, 'r') as f:
        data = csv.reader(f)
        wavenumber, intensity = [], []
        for wn, i in data:
            wavenumber.append(float(wn))
            intensity.append(float(i))

    wavenumber = numpy.array(wavenumber)
    intensity = numpy.array(intensity)

    return wavenumber, intensity


def find_peaks(data):
    """Find the Peaks, returns indices of peak centers."""
    return signal.find_peaks_cwt(data, numpy.arange(1, 30))


def fit_curve(xdata, ydata, **fitparams):
    """Fit the data to the equation using optional fit parameters."""
    return optimize.curve_fit(fit_eqn, xdata, ydata, **fitparams)


def do_the_fit(fname):
    print("Extracting Ddata from file: '{}'".format(fname))

    wavenumber, intensity = read_data(fname)

    # Remove data DC offset
    intensity = intensity - min(intensity)

    # Find the peaks of the data
    peakind = find_peaks(intensity)

    # Remove any peaks that are below the threshold
    peakind = [p for p in peakind if intensity[p] > INTENSITY_PEAK_THRESHOLD]

    print("Using {} peaks at indices: {}".format(len(peakind), peakind))

    print("Peak Intensities: {}".format(intensity[peakind]))

    the_fit_fn = fit_n_curves(len(peakind))

    # Get the fit guess
    fit_guess = build_fit_guess(wavenumber, intensity, peakind)

    # Do the fit
    try:

        pfit, pcov = optimize.curve_fit(
            the_fit_fn,
            wavenumber,
            intensity,
            p0=fit_guess,
            maxfev=30000)
    except RuntimeError as e:

        print("Fitting error: {}".format(str(e)))

    else:
        return wavenumber, intensity, peakind, pfit, pcov


def split_pfit(pfit, n):
    results = []

    l = len(pfit)
    assert l == n * 4

    for ndx in range(0, l, 4):
        results.append(pfit[ndx:min(ndx + 4, l)])

    return results


def plot_data_and_fits(wavenumber, intensity, pfit, peakind, title=None):

    plt.plot(wavenumber, intensity, label="Data")

    cum_intensity = numpy.zeros(len(intensity))

    for fit_params in split_pfit(pfit, len(peakind)):
        i_fit = numpy.array([fit_eqn(w, *fit_params) for w in wavenumber])
        cum_intensity += i_fit

        plt.plot(wavenumber, i_fit, label="{:.2f}".format(fit_params[2]))

    plt.plot(wavenumber, cum_intensity, "--", label="Cumulative Fit")

    plt.xlabel("Wavenumber")
    if title is not None:
        plt.title(title)
    plt.legend()


def main(fname):
    wavenumber, intensity, peakind, pfit, pcov = do_the_fit(fname)

    plot_data_and_fits(wavenumber, intensity, pfit, peakind, title=fname)

