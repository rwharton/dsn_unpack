import numpy as np
import sys
import struct
import os
import glob
import time
import datetime as dt
import os.path
import gdscc_params as par


def single_raw2fil(rawfile, outdir, source, tag, dateStr):
    tsamp    = par.TSAMP * 1.0e-6 # convert usec to sec
    nsamp    = par.PACKETS_PER_FILE * par.SAMPS_PER_PACKET
    chfreq   = par.LO
    chlabel  = "xyz"
    nskip    = 0

    num = int(rawfile.split('psr_recording_')[1].split('_')[0])
    t_off = num * nsamp * tsamp 
    mjd_off = t_off / (24. * 3600.)

    print('processing %s ...' % (rawfile))
    # write headers
    mjd = 57427.04
    
    # Calc mjd and add offset
    mjd =  calc_mjd2(dateStr)
    mjd += mjd_off 
    
    print("main: dateStr mjd = ", dateStr, mjd)

    # set the names of the filterbank files using the first dada filename
    #ofn = odir + "/" + tag + ".fil"
    ofn = "%s/%s-%04d.fil" %(outdir, tag, num)

    write_header(ofn, chfreq+par.BW, source, tsamp, mjd)

    ofp = open(ofn, "ab")

    # Write to fil
    packets_to_fil(rawfile, ofp)

    ofp.close()

    return


def raw2fil(idir, odir, source, tag, dateStr, pattern, nproc):
    tsamp    = par.TSAMP * 1.0e-6 # convert usec to sec
    chfreq   = par.LO
    chlabel  = "xyz"
    nskip    = 0

    filePattern = idir + "/" + pattern
    files = sorted(glob.iglob(filePattern))
    first = 1

    nfiles = len(files)

    fileNo = []
    for fn in files:
        no = int(fn.split('psr_recording_')[1].split('_')[0])
        fileNo.append(no)

    fileNoNp = np.asarray(fileNo)
    indxSorted = np.argsort(fileNoNp)

    files2 = []
    for i in range(nfiles):
        indx = indxSorted[i]
        f2 = files[indx]
        files2.append(f2)
        
    print("found these files to process (matched pattern)...")
    for fn in files2:
        print('>>> %s ...' % (fn))

    nfiles_processed = 0

    for fn in files2:
        print('processing %s ...' % (fn))
        if first:
            # write headers
            bname = "null"
            mjd = 57427.04

            mjd =  calc_mjd2(dateStr)
            print("main: dateStr mjd = ", dateStr, mjd)

            bname = tag + "-" + bname

            # set the names of the filterbank files using the first dada filename
            ofn = odir + "/" + tag + ".fil"

            write_header(ofn, chfreq+par.BW, source, tsamp, mjd)

            ofp = open(ofn, "ab")

        # Write to fil
        packets_to_fil(fn, ofp)

        nfiles_processed = nfiles_processed + 1
        first = 0
        if (nproc != 0) and (nproc == nfiles_processed):  # process only NPROC files
            break

    ofp.close()

    #t1 = time.time()
    #total = t1-t0
    #print "%s: time = %f", (myname,total)

    return


def read_packet_orig(fp, pxx):
    myname = "read_packet"
    valid = 1
    discard  = np.fromfile(fp, dtype=np.dtype('<Q'), count=1)
    discard  = np.fromfile(fp, dtype=np.dtype('<Q'), count=1)
    data_all = np.fromfile(fp, dtype=np.dtype('<B'), count = 2*4*512*2)
    if (len(data_all) <= 0):
        if (par.DEBUG):
            print("%s: number of packets read = %d" %(myname, nread))
        valid = 0
        return valid, pxx, kxx

    # data has the shape of 2x4x512x2 array
    # in order from left to right the indices refer to:
    # 2 samples per packet
    # 4,512 = 2x1024 (power and kurtosis)
    # 2 bytes per sample (shorts)
    data_all_rs = np.reshape(data_all, (2,4,512,2), order='F')
    #print(data_all_rs.dtype)
    data_all_packed=data_all_rs[1,:,:,:] + data_all_rs[0,:,:,:]*2**8;
    data_all_permute1=np.transpose(data_all_packed,(0,1,2));
    #print(data_all_permute1.dtype)

    pow_all = data_all_permute1[0,:,:];
    pow_all = np.append(pow_all, data_all_permute1[1,:,:])
    pow_all = np.reshape(pow_all, (2,512,2), order='C')
    #print(pow_all.dtype)

    #kurt_all=data_all_permute1[2,:,:];
    #kurt_all = np.append(kurt_all, data_all_permute1[3,:,:])
    #kurt_all = np.reshape(kurt_all, (2,512,2), order='C')

    pow_rs = pow_all[0,:,:];
    pow_rs = np.append(pow_rs, pow_all[1,:,:])
    pow_rs = np.reshape(pow_rs, (1024,2), order='C')
    #print(pow_rs.dtype)

    pxx[0,:] = pow_rs[:,0]
    pxx[1,:] = pow_rs[:,1]
    #print(pxx.dtype)

    #print(pxx.dtype)

    return valid, pxx


def read_packet(fp, pxx):
    myname = "read_packet"
    valid = 1
    #data_all = np.fromfile(fp, offset=16, dtype=np.dtype('<H'), count = 2*4*512)
    data_all = np.fromfile(fp, offset=16, dtype=np.dtype('>H'), count = 2*4*512)
    if (len(data_all) <= 0):
        if (par.DEBUG):
            print("%s: number of packets read = %d" %(myname, nread))
        valid = 0
        return valid, pxx, kxx

    pxx = unpack_data(data_all, pxx)

    return valid, pxx


def unpack_data(data_all, pxx):
    # data has the shape of 2x4x512 array
    # in order from left to right the indices refer to:
    # 2 samples per packet
    # 4,512 = 2x1024 (power and kurtosis)
    # 2 bytes per sample (shorts)
    
    dd = np.reshape(data_all, (2, 512, 4))
    dd = np.reshape(dd[:, :, 0:2].T, (1024, 2)).T

    pxx[:] = dd[:].astype('float32')

    return pxx


def unpack_rawfile(infile):
    # data organized as 16 header bytes followed by 
    # 8096 data bytes  

    t0_start = time.time()
    # Read in data as 16 bit ints
    dat = np.fromfile(infile, dtype='>H')
    t0_stop = time.time()
    dt0 = t0_stop - t0_start
    print("Reading: %.1f sec" %dt0)

    t1_start = time.time()
    # Reshape (8 2 byte ints = 16 hdr bytes)
    dat = np.reshape(dat, (-1, 2 * 512 * 4 + 8))

    # Now slice off the first header bytes
    dat = dat[:, 8 : 8 + 2 * 512 * 4]

    # Now reshape 
    dat = np.reshape(dat, (-1, 2, 512, 4))

    # Just grab the power channels
    dat = dat[:, :, :, 0:2]

    #print(dat.shape)

    t1_stop = time.time()
    dt1 = t1_stop - t1_start 
    print("Initial slicing: %.1f" %dt1)
   
    t2_start = time.time() 
    odat = np.zeros((dat.shape[0]*2, 1024), dtype='float32')

    #print(odat.shape)

    for ii in range(len(dat)):
        dd_ii = dat[ii]
        dd_ii = np.reshape(dd_ii.T, (1024, 2)).T
        odat[2*ii : 2*(ii+1)] = dd_ii

    t2_stop = time.time()
    dt2 = t2_stop - t2_start
    print("Reshaping and filling out array: %.1f sec" %dt2)
        
    return odat


def packets_to_fil_orig(ifn, ofp):
    myname = "packets_to_fil"
    npackets = par.PACKETS_PER_FILE;

    ifp = open(ifn,'rb');
    pxx2 = np.zeros((2, par.NCH), dtype=np.float32); # 2 samples

    for i in range(npackets):
        flag, pxx2 = read_packet(ifp, pxx2)
        if (flag == 0):
            print("%s: no packets to read = %d" %(myname, i))
            break
        if i == 0:
            print(pxx2[0,:])
            print(pxx2[0,:].dtype)
            print(len(pxx2[0, :]))
        # flip 1d array for proper dedisp
        write_data(ofp, pxx2[0, :][::-1])

        # flip 1d array for proper dedisp
        write_data(ofp, pxx2[1, :][::-1])

    ifp.close()
    return


def packets_to_fil(ifn, ofp):
    myname = "packets_to_fil"
    npackets = par.PACKETS_PER_FILE;

    # Read in raw data file 
    dat = unpack_rawfile(ifn)

    # Flip channels 
    dat = dat[:, ::-1]

    #print(dat.flags)

    # Write to output file
    tstart = time.time()
    ofp.write( dat.ravel() )
    tstop = time.time()
    dt = tstop - tstart
    print("Writing: %.1f sec" %dt)

    return


def send_string(ofp, str):
    ofp.write(struct.pack("<i",len(str)))
    ofp.write(str.encode('utf-8'))

def send_int(ofp, str, value):
    send_string(ofp, str)
    ofp.write(struct.pack("<i",value))

def send_double(ofp, str, value):
    send_string(ofp, str)
    ofp.write(struct.pack("<d",value))
    
def calc_mjd2(datestr):
    # example: 150720 17 40 00
    year  = int(datestr[0:2]) + 2000
    month = int(datestr[2:4])
    day   = int(datestr[4:6])

    a = (14 - month)/12
    y = year + 4800 - a
    m = month + 12 * a - 3

    jdn = day + (153 * m + 2)/5 + 365 * y +y/4 - y/100 + y/400 - 32045
    mjd = jdn - 2400000.5

    hh=datestr[7:9]; mm=datestr[10:12]; ss=datestr[13:15]
    hh=float(hh); mm=float(mm); ss=float(ss);
    mjd_frac = (hh + (mm/60.0) + (ss/3600.0))/24.0
    mjd = int(mjd) + mjd_frac

    return mjd


def write_header(ofn, fch1, source_name, tsamp, mjd):
    myname = "write_header"
    if (par.DEBUG):
        print("%s: ofn: %s" %(myname, ofn))
        
    ofp = open(ofn, "wb")
    send_string(ofp, "HEADER_START")

    send_string(ofp, "rawdatafile")
    send_string(ofp, ofn)

    send_string(ofp, "source_name")
    send_string(ofp, source_name)

    send_int(ofp, "telescope_id", par.TELESCOPE_ID)
    send_int(ofp, "machine_id", par.MACHINE_ID)
    send_int(ofp, "data_type", par.DATA_TYPE)
    send_double(ofp, "fch1", fch1)
    send_double(ofp, "foff", par.FOFF)
    send_int(ofp, "nchans", par.NCHANS)
    send_int(ofp, "nbits", par.OBITS)
    send_double (ofp, "tstart", mjd) 
    send_double(ofp, "tsamp", tsamp)
    send_int(ofp, "nifs", par.NIFS)
    send_int(ofp, "barycentric", par.BARYCENTRIC)

    send_string(ofp, "HEADER_END")

    ofp.close()


def write_data_orig(ofp, data):
    fmt = '<' + str(len(data)) + 'f'
    obuf = struct.pack(fmt, *data)
    ofp.write(obuf)


def write_data(ofp, data):
    ofp.write( data.tobytes() )


def get_scaninfo(scanfile, scanno):
    """
    Get info on scan number scanno from scanfile
    """
    scan = -1 
    name = 'None'
    dur_str = '-999'
    date_str = '-999'

    with open(scanfile, 'r') as fin:
        for line in fin:
            if line[0] in ["#", '\n']:
                continue
            else: pass

            cols = line.split()
            if len(cols) != 8:
                print("Scan file has wrong number of columns!")
                return
            else: pass

            if int(cols[7]) != scanno:
                continue
            else: pass

            scan     = int(cols[7])
            name     = cols[0].strip()
            dur_str  = ' '.join(cols[1:3])
            date_str = ' '.join(cols[3:7])

    print("scan number :  %d" %(scan))
    print("source name :  %s" %(name))
    print("duration    :  %s" %(dur_str))
    print("date string :  %s" %(date_str))

    if scan == -1:
        print("Scan %d not found!" %(scanno))
        return
    else:
        pass

    return scan, name, date_str


def make_output_dir(outdir, src_name, scan):
    """
    Make output scan dir if it doesnt exist already
    """
    odir = "%s/s%02d-%s" %(outdir, scan, src_name)
    if not os.path.exists(odir):
        os.mkdir(odir)
    return odir


def get_scan_table(indir):
    """
    Check for scan table in indir

    Assume form 'scan.table.[exp code]'
    
    Return path
    """
    st_pattern = '%s/scan.table.*' %(indir)
    stf = glob.glob(st_pattern)
    if len(stf) == 0:
        print("No scan table found in %s" %indir)
        sys.exit(0)
    elif len(stf) > 1:
        print("Multiple scan files found!")
        sys.exit(0)
    else:
        print("Found Scan Table: %s" %stf[0])

    return stf[0]


def check_input():
    """
    Check then return input vals

    Print usage if things not right
    """
    args = sys.argv[1:]

    if len(args) < 3:
        usage = "\nproc_1ch_gdscc.py scan_num indir outdir [nproc]"
        print(usage)
        help_txt = "\n" +\
                   "scan_num = Scan number to process from scan table file\n" +\
                   "indir    = input directory that has raw data files and scan table\n" +\
                   "outdir   = top of output directory (will make sub dirs for each scan)\n" +\
                   "nproc    = number of files to process (optional, default=0 means all)\n"
        print(help_txt)
        sys.exit(0)
    else: pass

    scanno   = int(args[0])
    indir    = args[1]
    outdir   = args[2]

    if len(args) == 3:
        nproc = 0
    elif len(args) == 4:
        nproc = int(args[3])

    return scanno, indir, outdir, nproc


def convert_one_scan(scanno, indir, outdir_top, nproc):
    """
    Convert raw files to fil for one scan
    """
    # Get scan file path
    scanfile = get_scan_table(indir)

    # Read scan file
    scan, name, date_str = get_scaninfo(scanfile, scanno)
    
    # Make dir for output of this scan
    outdir = make_output_dir(outdir_top, name, scan)
    
    # Make glob pattern
    pattern = "%s*_%03d" %(par.INBASE, scan)
    print("Matching pattern: %s" %(pattern))

    # Do the conversion
    raw2fil(indir, outdir, name, name, date_str, pattern, nproc)

    return
     

def convert_one_file(rawfile, scanno, indir, outdir_top):
    """
    Convert raw files to fil for one scan
    """
    tstart = time.time()

    # Get scan file path
    scanfile = get_scan_table(indir)

    # Read scan file
    scan, name, date_str = get_scaninfo(scanfile, scanno)
    
    # Make dir for output of this scan if it doesn't exist
    outdir = make_output_dir(outdir_top, name, scan)
    
    # Do the conversion
    #raw2fil(indir, outdir, name, name, date_str, pattern, nproc)
    single_raw2fil(rawfile, outdir, name, name, date_str)

    tstop = time.time()
    dt = tstop - tstart

    print("Took %.1f sec" %dt)

    return
     


debug = 0

if __name__ == '__main__':
    if debug:
        pass

    else:
        # Get input and check that it makes sense
        args = check_input()
        scanno, indir, outdir_top, nproc = args

        tstart = time.time() 

        convert_one_scan(scanno, indir, outdir_top, nproc)

        tstop = time.time()
        dt = tstop - tstart

        print("Took %.1f sec" %dt)
