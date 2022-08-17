# parameter file for running psr in incoherent mode

PACKETS_PER_FILE    = 23*2**16    # number of packets per file
SAMPS_PER_PACKET    = 2       # number of samples per packet 
                               # (not used in code...)

BW    = 640.0    # total bandwidth (MHz)
ACC   = 64       # number of accumulations per spectra
NCH   = 1024     # number of spectral channels
TSAMP = 102.4    # time sample (microseconds)
LO    = 8100.0   # LO Freq (MHz)

TELESCOPE_ID = 13
MACHINE_ID = 999

DATA_TYPE = 1 # filterbank
NCHANS = 1024
OBITS = 32
BW = 640.0
FOFF = -BW / NCHANS
NIFS = 1
BARYCENTRIC = 0
DEBUG = 1

INBASE = 'psr_recording'  # Base name of input files
