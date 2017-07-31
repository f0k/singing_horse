#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converts between audio files and spectrograms as .png or .npy.

For usage information, call without any parameters.

Author: Jan Schlüter
"""

import sys
import os
import subprocess
from PIL import Image

import numpy as np
try:
    from pyfftw.builders import rfft as rfft_builder
except ImportError:
    def rfft_builder(*args, **kwargs):
        return np.fft.rfft

# default values for some of the functions
SAMPLE_RATE = 22050
FRAME_LEN = 1024
FPS = 70
MEL_BANDS = 80
MIN_FREQ = 27.5
MAX_FREQ = 8000
VMIN = -14
VMAX = .5

def print_usage():
    print 'Converts between audio files and spectrograms as .png or .npy.'
    print 'Usage: %s INFILE OUTFILE' % sys.argv[0]
    print '  INFILE: audio file, .npy spectrogram or .png spectrogram'
    print '  OUTFILE: audio file, .npy spectrogram or .png spectrogram'



def read_ffmpeg(infile, sample_rate=SAMPLE_RATE, cmd='ffmpeg'):
    """
    Decodes a given audio file using ffmpeg, resampled to a given sample rate,
    downmixed to mono, and converted to float32 samples. Returns a numpy array.
    """
    call = [cmd, "-v", "quiet", "-i", infile, "-f", "f32le",
            "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
    samples = subprocess.check_output(call)
    return np.frombuffer(samples, dtype=np.float32)

def write_ffmpeg(samples, sample_rate=SAMPLE_RATE, outfile=None, cmd='ffmpeg'):
    """
    Encodes given samples using ffmpeg, in mp3 format. Either writes a file
    or returns raw data as a string.
    """
    if outfile is None:
        outfile = "pipe:1"
    call = [cmd, "-v", "quiet", "-y",
            "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "-i", "pipe:0",
            "-f", "mp3", outfile]
    process = subprocess.Popen(call, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output, err = process.communicate(samples.data)
    retcode = process.poll()
    if retcode:
        raise subprocess.CalledProcessError(retcode, call, output=output)
    if outfile == "pipe:1":
        return output



def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq,
                          max_freq, crop=False):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples. If `crop` is true-ish, crops
    the filterbank matrix above the bin corresponding to `max_freq`.
    """
    # prepare output matrix
    input_bins = (frame_len // 2) + 1
    filterbank = np.zeros((input_bins, num_bands))

    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    spacing = (max_mel - min_mel) / (num_bands + 1)
    peaks_mel = min_mel + np.arange(num_bands + 2) * spacing
    peaks_hz = 700 * (np.exp(peaks_mel / 1127) - 1)
    fft_freqs = np.linspace(0, sample_rate / 2., input_bins)
    peaks_bin = np.searchsorted(fft_freqs, peaks_hz)

    # fill output matrix with triangular filters
    for b, filt in enumerate(filterbank.T):
        # The triangle starts at the previous filter's peak (peaks_freq[b]),
        # has its maximum at peaks_freq[b+1] and ends at peaks_freq[b+2].
        left_hz, top_hz, right_hz = peaks_hz[b:b+3]  # b, b+1, b+2
        left_bin, top_bin, right_bin = peaks_bin[b:b+3]
        # Create triangular filter compatible to yaafe
        filt[left_bin:top_bin] = ((fft_freqs[left_bin:top_bin] - left_hz) /
                                  (top_hz - left_hz))
        filt[top_bin:right_bin] = ((right_hz - fft_freqs[top_bin:right_bin]) /
                                   (right_hz - top_hz))
        filt[left_bin:right_bin] *= 2 / (right_hz - left_hz)
        #filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()  # better, but not what yaafe does
    if crop:
        filterbank = filterbank[:peaks_bin[-1]]
    return filterbank

def invert_filterbank(filterbank, method):
    """
    Inverts a given filterbank either by computing its pseudo-inverse
    (for ``method="pinv"``) or its transpose (for ``method="pinv"``).
    """
    if method == 'pinv':
        return np.linalg.pinv(filterbank)  # pseudo-inverse
    elif method == 'transpose':
        # we transpose the mel bank and undo the per-band normalizations
        scales = filterbank.sum(axis=1)
        # scales are nicely defined only where there are overlapping filters.
        # we do a linear expansion outside that range.
        bottom = np.where(scales)[0][0]
        scales[:bottom] = scales[bottom] + (scales[bottom] - scales[bottom+1]) * np.arange(bottom)[::-1]
        top = np.where(scales)[0][-1]
        scales[top:] = scales[top-1] + (scales[top-1] - scales[top-2]) * np.arange(len(scales) - top)
        np.maximum(scales, 1e-16, scales)  # clip values that are too small
        # now we can return a transposed melbank that inverts the normalizations
        return filterbank.T / scales**2
    else:
        raise ValueError("Unsupported mel filterbank inversion method: %s" % method)



def logarithmize(spect):
    """Computes logarithmic magnitudes in-place."""
    eps = 1e-7
    np.maximum(spect, eps, spect)
    np.log(spect, spect)

def undo_logarithmize(spect, inplace=False):
    """Converts logarithmic magnitudes back to linear magnitudes."""
    if inplace:
        np.exp(spect, spect)
    else:
        return np.exp(spect)



def filtered_stft(samples, frame_len, hop_size, filterbank):
    """
    Computes an STFT, applying a filterbank on the way to minimize memory use.
    """
    window = np.hanning(frame_len)
    rfft = rfft_builder(samples[:frame_len], n=frame_len)
    spect = np.vstack(np.dot(np.abs(rfft(samples[pos:pos+frame_len] * window))[:len(filterbank)],
                            filterbank)
            for pos in range(0, len(samples) - frame_len + 1, hop_size))
    return spect

def undo_melfilter(spect, sample_rate=SAMPLE_RATE, frame_len=FRAME_LEN, min_freq=MIN_FREQ, max_freq=MAX_FREQ, method='transpose'):
    """
    Converts a mel spectrogram into a linear-frequency spectrogram.
    """
    num_frames, num_bands = spect.shape
    melbank = create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq, crop=True)
    spect = np.dot(spect, invert_filterbank(melbank, method))
    return spect

def undo_stft(spect, hop_size, frame_len=None, unwindow='auto'):
    """
    Undoes an SFTF via overlap-add, returning a numpy array of samples.
    """
    # transform into time domain
    spect = np.fft.irfft(spect, n=frame_len, axis=1)
    # overlap-and-add
    num_frames, frame_len = spect.shape
    win = np.hanning(frame_len)
    #win = np.sin(np.pi * np.arange(frame_len) / frame_len)
    #win = 1
    if unwindow == 'auto':
        unwindow = (hop_size <= frame_len//2)
    samples = np.zeros((num_frames - 1) * hop_size + frame_len)
    if unwindow:
        factors = np.zeros_like(samples)
    for idx, frame in enumerate(spect):
        oidx = int(idx*hop_size)
        samples[oidx:oidx+frame_len] += frame * win
        if unwindow:
            factors[oidx:oidx+frame_len] += win**2
    if unwindow:
        np.maximum(factors, .1 * factors.max(), factors)
        samples /= factors
    return samples



def extract_melspect(samples_or_file, sample_rate=SAMPLE_RATE, frame_len=FRAME_LEN, num_bands=MEL_BANDS, min_freq=MIN_FREQ, max_freq=MAX_FREQ, fps=FPS):
    """
    Computes a mel spectrogram for a given input file or numpy array of samples.
    """
    # read input samples (if they're not samples already)
    if not isinstance(samples_or_file, np.ndarray):
        try:
            samples = read_ffmpeg(samples_or_file, sample_rate)
        except OSError:
            samples = read_ffmpeg(samples_or_file, sample_rate, 'avconv')
    else:
        samples = samples_or_file
    # apply STFTs and mel bank and logarithmize
    melbank = create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq, crop=True)
    hop_size = int(sample_rate / fps + 0.5)
    spect = filtered_stft(samples, frame_len, hop_size, melbank)
    logarithmize(spect)
    return spect.astype(np.float32)

def undo_melspect(spect, sample_rate=SAMPLE_RATE, fps=FPS, frame_len=FRAME_LEN, min_freq=MIN_FREQ, max_freq=MAX_FREQ, invert_melbank_method='transpose', phases='random', normalize=False):
    """
    Resynthesizes a mel spectrogram into a numpy array of samples.
    """
    # undo logarithmic scaling
    spect = undo_logarithmize(spect)
    # undo Mel filterbank
    spect = undo_melfilter(spect, sample_rate, frame_len, min_freq, max_freq, invert_melbank_method)
    # randomize or reuse phases
    if phases == 'random':
        spect = spect * np.exp(np.pi*2.j*np.random.random(spect.shape))
    elif phases is not None:
        spect = spect * np.exp(1.j * np.angle(phases))
    # undo STFT
    hop_size = sample_rate / fps
    samples = undo_stft(spect, hop_size, frame_len)
    # normalize if needed
    if normalize:
        samples -= samples.mean()
        samples /= np.abs(samples).max()
    return samples.astype(np.float32)


def load_png(infile, vmin=VMIN, vmax=VMAX):
    img = Image.open(infile)
    data = np.asarray(img)[::-1].T.astype(np.float32)
    data = data / 255 * (vmax - vmin) + vmin
    return data

def write_png(outfile, data, vmin=VMIN, vmax=VMAX):
    data = np.clip((data - vmin) / (vmax - vmin), 0, 1) * 255
    img = Image.fromarray(data.T[::-1].astype(np.uint8))
    img.save(outfile)


def main():
    if len(sys.argv) < 3:
        print_usage()
        return

    # 'parse' command line
    infile, outfile = sys.argv[1:]

    # read input
    if infile.endswith('.npy'):
        spect = np.load(infile)
    elif infile.endswith('.png'):
        spect = load_png(infile)
    else:
        spect = extract_melspect(infile)

    # write output
    if outfile.endswith('.npy'):
        np.save(outfile, spect)
    elif outfile.endswith('.png'):
        write_png(outfile, spect)
    else:
        samples = undo_melspect(spect, normalize=False)
        try:
            write_ffmpeg(samples, sample_rate=SAMPLE_RATE, outfile=outfile)
        except OSError:
            write_ffmpeg(samples, sample_rate=SAMPLE_RATE, outfile=outfile, cmd='avconv')

if __name__=="__main__":
    main()

