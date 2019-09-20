#!/usr/bin/env python

"""A working (and used!) example of using pyinotify to trigger data reduction.

A multiprocessing pool is used to perform the reduction in
an asynchronous and parallel fashion.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import obspy
from StringIO import StringIO
from surf_seis import vibbox as vibbox
from surf_seis.phasepapy.phasepicker import aicdpicker
import sys
from pyproj import Proj, transform
import multiprocessing
import logging, logging.handlers
import shutil
import pyinotify
import uuid
import subprocess

logger = logging.getLogger()

class options():
    """Dummy class serving as a placeholder for argparse handling."""
    watch_dir = "/data1/vbox/incoming/"
    log_dir = "/data1/vbox/log"
    output_dir = "/data1/vbox/output"
    output_dir_trig = "/home/sigmav/sigmav_ext"
    ext_dir = "/home/sigmav/sigmav_ext"

    nthreads = 6

class RsyncNewFileHandler(pyinotify.ProcessEvent):
    """Identifies new rsync'ed files and passes their path for processing.

    rsync creates temporary files with a `.` prefix and random 6 letter suffix,
    then renames these to the original filename when the transfer is complete.
    To reliably catch (only) new transfers while coping with this file-shuffling,
    we must do a little bit of tedious file tracking, using
    the internal dict `tempfiles`.
    Note we only track those files satisfying the condition
    ``file_predicate(basename)==True``.

    """
    def my_init(self, nthreads, file_predicate, file_processor):
        self.mask = pyinotify.IN_MOVED_TO | pyinotify.IN_CLOSE_WRITE
        self.predicate = file_predicate
        self.process = file_processor

    def process_IN_MOVED_TO(self, event):
        #Now rsync has renamed the file to drop the temporary suffix.
        #NB event.name == basename(event.pathname) AFAICT
        logger.info('IN_MOVED_TO Sending for processing: %s', event.pathname)
#        check_external_harddrives()
        self.process(event.pathname)

    def process_IN_CLOSE_WRITE(self, event):
        #Now rsync has renamed the file to drop the temporary suffix.
        #NB event.name == basename(event.pathname) AFAICT
        logger.info('IN_CLOSE_WRITE Sending for processing: %s', event.pathname)
#        check_external_harddrives()
        self.process(event.pathname)

def process_file_trigger(fname):
    try:
        st = vibbox.vibbox_read(fname)
        logger.info('Read ' + fname)
    except Exception as e:
        logger.info(e)
        logger.info('Cannot read ' + fname)
    try:
        st = vibbox.vibbox_preprocess(st)
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in preprocessing ' + fname)
    try:
        new_triggers = vibbox.vibbox_trigger(st.copy(), num=20)
        # clean triggers
        new_triggers = vibbox.vibbox_checktriggers(new_triggers, st.copy())
        res = new_triggers.to_csv(header=False, float_format='%5.4f', index=False)
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in triggering ' + fname)
        return 0
    try:
        st = vibbox.vibbox_custom_filter(st)
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in filtering ' + fname)
    try:
        for index, ev in new_triggers.iterrows():
            ste = st.copy().trim(starttime = ev['time'] - 0.01,  endtime = ev['time'] + ev['duration'] + 0.01)
            outname = mseedpath + '{:10.2f}'.format(ev['time'].timestamp)  + '.mseed'
            if not os.path.exists(mseedpath):
                os.makedirs(mseedpath)
            ste.write(outname, format='mseed')
    except Exception as e:
        logger.info(e)
        logger.info('Error in saving ' + fname)
        return 0
    return new_triggers

def process_file_pick(trigger):
    # calculate trigger strength on Accelerometers and skip picking if it is small
    trigger_strength = np.prod(trigger[26:62])
    #logger.debug('Trigger strength: ' + str(trigger_strength))
    if trigger_strength < 10000:
        return 0
    # check if we trigger at a minimum of 4 accelerometers
    trig_at_station = np.zeros(len(stations))
    triggering_stations = []
    for ii in np.arange(2,62):
        my_station = trigger.index[ii][3:-5]
        if trigger[ii]>1.3:
            trig_at_station[stations.index(my_station)] = 1
            if not my_station in triggering_stations:
                triggering_stations.append(my_station)
    if np.sum(trig_at_station[24:]) <= 4:
        return 0
    fname = mseedpath + '{:10.2f}'.format(trigger['time'].timestamp)  + '.mseed'

    try:
        logger.info('Reading ' + fname)
        ste = obspy.read(fname)
    except Exception as e:
        logger.info(e)
        logger.info('Error while reading ' + fname)
        return 0
    try:
        starttime = ste[0].stats.starttime
        endtime = ste[0].stats.endtime
    except Exception as e:
        logger.debug(e)
        logger.debug('Cannot determine start or endtime: ' + fname)
        return 0
    try:
        ste = ste.trim(starttime=starttime, endtime=endtime - 0.01)
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in trimming ' + fname)
        return 0
    try:
        picks = pick_event(ste, triggering_stations, trigger['time'])
        if len(picks) > 0:
            logger.debug(fname + ' picked')
            logger.debug(picks)
        else:
            logger.debug('No picks in ' + fname)
            return 0
#        q.put('#' + picks)
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in picking ' + fname)
        return 0
    try:
        if len(picks) > 300:
            plot_picks(picks, ste)
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in plotting ' + fname)
        return 0
    return picks

# def plot_picks(my_picks, st):
#     colspecs = [(0, 14), (14, 42), (42, 51), (51, 79), (79, 80), (80, 87)]
#     names = ['eventid', 'origin', 'station', 'pick', 'phase', 'snr']
#     my_picks = pd.read_fwf(StringIO(my_picks), colspecs=colspecs, names=names)
#     my_picks['origin'] = my_picks['origin'].apply(obspy.UTCDateTime)
#     my_picks['pick'] = my_picks['pick'].apply(obspy.UTCDateTime)
# 
#     num_picks= len(my_picks)
#     starttime = st[0].stats.starttime
#     dt = st[0].stats.delta
#     fig, axs = plt.subplots(num_picks, 1, sharex=True)
#     plt.suptitle('event ' + str(starttime), color='red')
#     for ii in range(num_picks):
#         station = my_picks['station'].iloc[ii]
#         tr = st.select(station=station)
#         axs[ii].plot(tr[0].data, c='k')
#         axs[ii].hold(True)
#         axs[ii].set_yticks([])
#         ix = (my_picks['pick'].iloc[ii] - starttime) / dt
#         axs[ii].axvline(x=ix, c=(1.0, 0.0, 0.0))
#         ylim = axs[ii].get_ylim()
#         axs[ii].text(0, (ylim[1] - ylim[0])* 0.9 + ylim[0], station)
#     fig.set_size_inches(10.5, 2*num_picks)
#     plt.tight_layout(pad=0.0, h_pad=0.0)
#     plt.savefig(pngpath + '{:10.2f}'.format(my_picks['eventid'].iloc[0]) + '.png')
#     plt.close()
#     plt.gcf().clear()
#     plt.cla()
#     plt.clf()

def plot_picks(my_picks, st):
    colspecs = [(0, 14), (14, 42), (42, 51), (51, 79), (79, 80), (80, 87)]
    names = ['eventid', 'origin', 'station', 'pick', 'phase', 'snr']
    my_picks = pd.read_fwf(StringIO(my_picks), colspecs=colspecs, names=names)
    my_picks['origin'] = my_picks['origin'].apply(obspy.UTCDateTime)
    my_picks['pick'] = my_picks['pick'].apply(obspy.UTCDateTime)

    num_picks= len(my_picks)
    starttime = st[0].stats.starttime
    dt = st[0].stats.delta
    fig, axs = plt.subplots(num_picks + 1, 1, sharex=True)
    plt.suptitle('event ' + str(starttime), color='red')
    for ii in range(num_picks):
        station = my_picks['station'].iloc[ii]
        tr = st.select(station=station)
        axs[ii].plot(tr[0].data, c='k')
        axs[ii].hold(True)
        axs[ii].set_yticks([])
        ix = (my_picks['pick'].iloc[ii] - starttime) / dt
        axs[ii].axvline(x=ix, c=(1.0, 0.0, 0.0))
        ylim = axs[ii].get_ylim()
        axs[ii].text(0, (ylim[1] - ylim[0])* 0.9 + ylim[0], station)
    # plot CASSM signal
    axs[ii+1].plot(st[60].data, c='k')
    axs[ii+1].hold(True)
    axs[ii+1].set_yticks([])
    axs[ii+1].text(0, (ylim[1] - ylim[0])* 0.9 + ylim[0], 'CASSM')
    fig.set_size_inches(10.5, 2*(num_picks+1))
    plt.tight_layout(pad=0.0, h_pad=0.0)
    if not os.path.exists(pngpath):
        os.makedirs(pngpath)
    plt.savefig(pngpath + '{:10.2f}'.format(my_picks['eventid'].iloc[0]) + '.png')
    plt.close()
    plt.gcf().clear()
    plt.cla()
    plt.clf()

def pick_event(ste, stations, time):
    Picker = aicdpicker.AICDPicker(t_ma = 0.001, nsigma = 8, t_up = 0.01, nr_len = 0.002, nr_coeff = 2, pol_len = 100, pol_coeff = 5, uncert_coeff = 3)
    output = StringIO()
    for sta in stations:
        if sta in hydrophones: # hydrophones
            station = sta.split('.')[0]
            tr = ste.select(station = station)[0]
            scnl, picks, polarity, snr, uncert = Picker.picks(tr)
            if len(picks) > 0:
                pickstring = '{:10.2f}'.format(time.timestamp) + ' ' + str(time) + ' ' + '{:8s}'.format(sta) + ' ' + str(picks[0]) + ' P ' + '{:5.0f}'.format(snr[0]) + '\n'
                output.write(pickstring)
        elif sta in accelerometers: # accelerometers
            got_pick = False
            station = sta.split('.')[0]
            tr = ste.select(station = station)
            picks = np.zeros(3)
            snr = np.zeros(3)
            for ii in range(3):
                scnl, t_picks, polarity, t_snr, uncert = Picker.picks(tr[ii])
                if len(t_picks)>0:
                    got_pick = True
                    picks[ii] = t_picks[0]
                    snr[ii] = t_snr[0]
            if got_pick:
                if max(picks) - min(picks) > 0.00003:
                    # select pick with largest SNR
                    picks = picks[np.argmax(snr)]
                    snr = np.max(snr)
                else:
                    picks = np.mean(picks)
                    snr = np.mean(snr)
                pickstring = '{:10.2f}'.format(time.timestamp) + ' ' + str(time) + ' ' + '{:8s}'.format(sta) + ' ' + str(obspy.UTCDateTime(picks)) + ' P ' + '{:5.0f}'.format(snr) + '\n'
                output.write(pickstring)
    return output.getvalue()

def generate_hyp_input(df_picks, my_uuid):
    fn_hypinput = output_path + my_uuid + '/' + 'hyp_input.dat'       # output for hypoinverse
    eventid_0 = 1                                    # set to 1 for new catalog or larger to continue catalog
    min_picks = 5
    only_accelerometers = False

    colspecs = [(0, 14), (14, 42), (42, 51), (51, 79), (79, 80), (80, 87)]
    names = ['eventid', 'origin', 'station', 'pick', 'phase', 'snr']
#    df_picks = pd.read_fwf(fn_picks, colspecs=colspecs, names=names)
    df_picks['origin'] = df_picks['origin'].apply(obspy.UTCDateTime)
    df_picks['pick'] = df_picks['pick'].apply(obspy.UTCDateTime)
    df_picks = df_picks[df_picks['station'] != 'CTrig']

    if only_accelerometers:
        df_picks = df_picks[df_picks['station'].str.contains('A')]
    eventid = np.unique(df_picks['eventid'])
    df_picks['eventid'] = df_picks['eventid'].replace(to_replace=eventid, value = np.arange(len(eventid)) + eventid_0)

    df_picks = df_picks.drop_duplicates(subset = ['eventid', 'station'], keep='first')
    logger.debug(str(len(eventid)) + ' events will be written')    # need to log triggers to file

    df_stations = pd.read_csv('/home/sigmav/Vibbox_processing/stations_local_coordinates.txt', delimiter='\t')
    df_stations['station'] = df_stations['station'].str.rstrip()
    df_stations['z'] = -df_stations['z']
    mean_x0 = df_stations['x'].mean()
    mean_y0 = df_stations['y'].mean()
    max_z0 = df_stations['z'].max()
    proj_WGS84 = Proj(init='epsg:4326') # WGS-84
    proj_utm13N = Proj(init='epsg:32613') # UTM 13N
    easting_0 = 598445
    northing_0 = 4912167
    depth_0 = 0
    # scale by a factor of 1000 (and from km to m)
    df_stations['easting_scaled'] = (df_stations['x'] - mean_x0) * 1e6 + easting_0
    df_stations['northing_scaled'] = (df_stations['y'] - mean_y0) * 1e6 + northing_0
    # scale by a factor of 1000 (stay with km units)
    # station elevations with positive is up
    df_stations['depth_scaled'] = (df_stations['z'] - max_z0) * 1e3 + depth_0
    df_stations['lon_scaled'], df_stations['lat_scaled'] = transform(proj_utm13N, proj_WGS84,
               df_stations['easting_scaled'].values, df_stations['northing_scaled'].values)
    df_stations['lon_deg'] = np.ceil(df_stations['lon_scaled'])
    df_stations['lon_min'] = abs(df_stations['lon_scaled'] - df_stations['lon_deg'])*60;
    df_stations['lat_deg'] = np.floor(df_stations['lat_scaled'])
    df_stations['lat_min'] = abs(df_stations['lat_scaled'] - df_stations['lat_deg'])*60;

    events = df_picks['eventid'].unique()

    try:
        shutil.copyfile('/home/sigmav/Vibbox_processing/surf.crh', output_path + my_uuid + '/surf.crh')
        shutil.copyfile('/home/sigmav/Vibbox_processing/hypinst', output_path + my_uuid + '/hypinst')
        shutil.copyfile('/home/sigmav/Vibbox_processing/hyp1.40_ms', output_path + my_uuid + '/hyp1.40_ms')
        shutil.copyfile('/home/sigmav/Vibbox_processing/hyp_stations.dat', output_path + my_uuid + '/hyp_stations.dat')
    except Exception as e:
        logger.info(e)
        logger.info('Cannot write to output directory ' + output_path + my_uuid)
        return 0
    phasefile = open(fn_hypinput, 'wb')
    try_to_locate = False
    for ev in events:
        picks = df_picks.loc[df_picks['eventid'] == ev]
        num_picks = len(picks)
        if num_picks <= min_picks:
            continue
        try_to_locate = True
        origin_time = picks['pick'].min()
        origin_station = picks['station'].loc[picks['pick'] == origin_time]
        origin_station = df_stations[df_stations['station'] == origin_station.values[0]]

        phasefile.write(
                    '{:4d}'.format(origin_time.year) +
                    '{:02d}'.format(origin_time.month) +
                    '{:02d}'.format(origin_time.day) +
                    '{:02d}'.format(origin_time.hour) +
                    '{:02d}'.format(origin_time.minute) +
                    '{:02d}'.format(origin_time.second) +
                    '{:02d}'.format(int(origin_time.microsecond/1e4))  +
                    '{:2.0f}'.format(float(origin_station.lat_scaled)) + ' ' +
                    '{:4.0f}'.format(float(origin_station.lat_scaled - int(origin_station.lat_scaled))*6000) +
                    '{:3d}'.format(int(abs(origin_station.lon_scaled))) + ' ' +
                    '{:4.0f}'.format(abs(float(origin_station.lon_scaled - int(origin_station.lon_scaled))*6000)) + ' ' +
                    '{:4.0f}'.format(float(origin_station.depth_scaled) * -100) +
                    '{:3.0f}'.format(float(0)) +
                    '{:107d}'.format(np.int(ev)) + '\n')
    #   phase lines
        for index, pick in picks.iterrows():
            traveltime_scaled = (pick['pick'] - origin_time) * 1000
            picktime_scaled = pick['pick'] + traveltime_scaled
            if pick['phase'] == 'P':
                phasefile.write('{:5s}'.format(pick.station) + 'SV       P 0' +
                                '{:04d}'.format(picktime_scaled.year) + '{:02d}'.format(picktime_scaled.month) + '{:02d}'.format(picktime_scaled.day) +
                                 '{:02d}'.format(picktime_scaled.hour) + '{:02d}'.format(picktime_scaled.minute) +
                                 '{:5.0f}'.format((picktime_scaled.second + picktime_scaled.microsecond/1e6) * 100) + '   0101    0   0' + '\n')
            if pick['phase'] == 'S':
                phasefile.write('{:5s}'.format(pick.station_compact) + 'SV         4' +
                                '{:04d}'.format(picktime_scaled.year) + '{:02d}'.format(picktime_scaled.month) + '{:02d}'.format(picktime_scaled.day) +
                                 '{:02d}'.format(picktime_scaled.hour) + '{:02d}'.format(picktime_scaled.minute) +
                                 '   0   0  0 ' + '{:5.0f}'.format((picktime_scaled.second + picktime_scaled.microsecond/1e6) * 100) + ' S 2\n')
        phasefile.write('                                                                ' + '{:6d}'.format(np.int(ev)) + '\n')
    phasefile.close()
    return try_to_locate

def run_hypoinverse(my_uuid):
    from subprocess import call
    root_dir = os.getcwd()
    os.chdir(output_path + my_uuid)
    call('./hyp1.40_ms')
    os.chdir(root_dir)

def postprocess_hyp(my_uuid):
    infile = output_path + my_uuid + '/hyp_output.csv'
    output = output_path + my_uuid + '/catalog.csv'
    colspecs = [(0, 24), (24, 31), (32, 42), (43, 48), (48, 62), (62, 64), (65, 68), (69, 74), (75, 81), (82, 87), (88, 92), (93, 95), (97, 108)]
    names = ['date_scaled', 'lat_scaled', 'lon_scaled', 'depth_scaled', 'mag', 'num', 'gap', 'dmin', 'rms', 'erh', 'erz', 'qasr', 'eventid']
    df_loc = pd.read_fwf(infile, colspecs=colspecs, names=names, skiprows=1)
    # check if any events have been located
    if len(df_loc)==0:
	return 0
    df_loc['date_scaled'] = df_loc['date_scaled'].apply(obspy.UTCDateTime)
    df_loc['date'] = 0

    df_stations = pd.read_csv('/home/sigmav/Vibbox_processing/stations_local_coordinates.txt', delimiter='\t')
    df_stations['station'] = df_stations['station'].str.rstrip()
    df_stations['z'] = -df_stations['z']
    mean_x0 = df_stations['x'].mean()
    mean_y0 = df_stations['y'].mean()
    max_z0 = df_stations['z'].max()
    # UTM 13T 598445 4912167
    proj_WGS84 = Proj(init='epsg:4326') # WGS-84
    proj_utm13N = Proj(init='epsg:32613') # UTM 13N
    easting_0 = 598445
    northing_0 = 4912167
    depth_0 = 0
    df_loc['easting_scaled'], df_loc['northing_scaled'] = transform(proj_WGS84, proj_utm13N,
               df_loc['lon_scaled'].values, df_loc['lat_scaled'].values)
    df_loc['x'] = ((df_loc['easting_scaled'] - easting_0)/1e6 + mean_x0) * 1000
    df_loc['y'] = ((df_loc['northing_scaled'] - northing_0)/1e6 + mean_y0) * 1000
    df_loc['z'] = ((df_loc['depth_scaled'] - depth_0) / 1000 - max_z0) * 1000

    df_picks = pd.read_csv(output_path + my_uuid + '/picks.csv')
    df_picks['origin'] = df_picks['origin'].apply(obspy.UTCDateTime)
    df_picks['pick'] = df_picks['pick'].apply(obspy.UTCDateTime)
    df_picks = df_picks[df_picks['station'] != 'CTrig']
    eventid = np.unique(df_picks['eventid'])
    df_picks['eventid'] = df_picks['eventid'].replace(to_replace=eventid, value = np.arange(len(eventid)) + 1)
    # rescale origin time
    for index, ev in df_loc.iterrows():
        picks = df_picks[df_picks['eventid'] == ev.eventid]
        min_pick = picks['pick'].min()
        date_scaled = df_loc['date_scaled'][df_loc['eventid'] == ev.eventid].tolist()[0]
        traveltime = (min_pick - date_scaled)
        df_loc['date'][df_loc['eventid'] == ev.eventid] = date_scaled + traveltime - traveltime/1000.0
    return df_loc

def process_rawfile(file):
    if not is_rawfile(file):
        return 'Finished processing file, not a raw file: ' + file
    try:
        with open(fn_recent) as f:
             recent_files = f.read().splitlines()
        if file in recent_files:
             logger.debug('File was already processed: ' + file)
             return 'Finished processing file, already processed: ' + file
        else:
             logger.debug('File not yet processed send ' + file)
             recent_files.append(file)
             if len(recent_files) > 100:
                 recent_files = recent_files[1:]
             with open(fn_recent, 'w') as f:
                 for item in recent_files:
                     f.write("%s\n" % item)
    except Exception as e:
        logger.debug(e)
        logger.debug('Cannot determine if in recent files ', file_path)

    my_uuid = str(uuid.uuid4())
    try:
        os.makedirs(output_path + my_uuid)
    except Exception as e:
        logger.info(e)
        logger.info('Cannot write to output directory ', output_path + my_uuid)
    df_triggers = process_file_trigger(file)
    filename = os.path.split(file)[-1]
    try:
# this may cause issues with some file systems, use subprocess.call instead
#        shutil.move(file, os.path.join(options.ext_dir, filename))
        subprocess.call(['mv',file, os.path.join(options.ext_dir, filename)])
        logger.info('Moved to external drive ' + os.path.join(options.ext_dir, filename))
    except Exception as e:
        logger.info(e)
        logger.info('Cannot move to external drive ' + os.path.join(options.ext_dir, filename))
    try:
        if len(df_triggers)>0:
            df_triggers.to_csv(fn_triggers, mode='a', header=False, index=False)
    except Exception as e:
        logger.info(e)
        logger.info('Cannot write triggers')
    try:
        colspecs = [(0, 14), (14, 42), (42, 51), (51, 79), (79, 80), (80, 87)]
        names = ['eventid', 'origin', 'station', 'pick', 'phase', 'snr']
        df_picks = pd.DataFrame()
        for index, trig in df_triggers.iterrows():
            new_picks = process_file_pick(trig)
            if new_picks != 0:
                df_new_picks = pd.read_fwf(StringIO(new_picks), colspecs=colspecs, names=names)
                df_picks = df_picks.append(df_new_picks)
        df_events = pd.DataFrame()
        if len(df_picks) > 0:
            df_picks.to_csv(output_path + my_uuid + '/picks.csv', index=False)
            df_picks.to_csv(fn_picks, header=False, index=False, mode='a')
    except Exception as e:
        logger.info(e)
        logger.info('Error in processing picks')
    try:
        if len(df_picks) > 0:
        # Compile scaled Hypoinverse input
            try_to_locate = generate_hyp_input(df_picks, my_uuid)
        # Run Hypoinverse
            if try_to_locate:
                run_hypoinverse(my_uuid)
            # Postprocess hypoinverse
                df_events = postprocess_hyp(my_uuid)
                df_events.to_csv(fn_catalog, header=False, mode='a', index=False)
    except Exception as e:
        logger.info(e)
        logger.info('Error during locating events')
    # cleanup
    try:
        if os.path.exists(output_path + my_uuid):
            shutil.rmtree(output_path + my_uuid)
    except:
        sys.exit('Cannot clean temp directory ', output_path + my_uuid)

    return 'Finished processing file ' + file + '.\n' + str(len(df_triggers)) + ' triggers found\n' + str(len(df_events)) + ' events located'

def is_rawfile(filename):
    """Predicate function for identifying incoming AMI data"""
    if '.dat' in filename:
        return True
    return False

def newfile_callback(file):
    """Used to log the 'job complete' / error message in the master thread."""
    logger.info('*** File submitted for processing: ' + file)
    # check_external_harddrives()

def processed_callback(summary):
    """Used to log the 'job complete' / error message in the master thread."""
    logger.info('*** Job complete: ' + summary)

def main(options):


    def simply_process_rawfile(file_path):
        """
        Wraps `process_rawfile` to take a single argument (file_path).

        This is the trivial, single threaded version -
        occasionally useful for debugging purposes.
        """
        newfile_callback(file_path)
        summary = process_rawfile(file_path)
        processed_callback(summary)

    def asynchronously_process_rawfile(file_path):
        """
        Wraps `process_rawfile` to take a single argument (file_path).

        This version runs 'process_rawfile' asynchronously via the pool.
        This provides parallel processing, at the cost of being harder to
        debug if anything goes wrong (see notes on exception catching above)
        """
        pool.apply_async(process_rawfile,
             [file_path],
             callback=processed_callback)

    """Define processing logic and fire up the watcher"""
    watch_dir = options.watch_dir
    try:
        pool = multiprocessing.Pool(options.nthreads)

        handler = RsyncNewFileHandler(nthreads=options.nthreads,
                                      file_predicate=is_rawfile,
                                      #file_processor=simply_process_rawfile
                                      file_processor=asynchronously_process_rawfile
                                     )
        wm = pyinotify.WatchManager()
        notifier = pyinotify.Notifier(wm, handler)
        wm.add_watch(options.watch_dir, handler.mask, rec=True)
        log_preamble(options)
        notifier.loop()
    finally:
        pool.close()
        pool.join()

    return 0

def log_preamble(options):
    logger.info("***********")
    logger.info('Watching %s', options.watch_dir)
    logger.info('Output dir %s', options.output_dir)
    logger.info('Log dir %s', options.log_dir)
    logger.info('External dir %s', options.ext_dir)
    logger.info("***********")

def setup_logging(options):
    """Set up basic (INFO level) and debug logfiles

    These should list successful reductions, and any errors encountered.
    We also copy the basic log to STDOUT, but it is expected that
    the monitor script will be daemonised / run in a screen in the background.
    """
    if not os.path.isdir(options.log_dir):
        os.makedirs(options.log_dir)
    log_filename = os.path.join(options.log_dir, 'autocruncher_log')
    date_fmt = "%a %d %H:%M:%S"
    std_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', date_fmt)
    debug_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s', date_fmt)

    info_logfile = logging.handlers.RotatingFileHandler(log_filename,
                            maxBytes=5e7, backupCount=10)
    info_logfile.setFormatter(std_formatter)
    info_logfile.setLevel(logging.INFO)
    debug_logfile = logging.handlers.RotatingFileHandler(log_filename + '.debug',
                            maxBytes=5e7, backupCount=10)
    debug_logfile.setFormatter(debug_formatter)
    debug_logfile.setLevel(logging.DEBUG)

    log_stream = logging.StreamHandler()
    log_stream.setFormatter(std_formatter)
    log_stream.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(info_logfile)
    logger.addHandler(log_stream)
    logger.addHandler(debug_logfile)

if __name__ == '__main__':
    output_path = options.output_dir + '/'
    output_path_trig = options.output_dir + '/'

    # output filenames
    my_datestr = obspy.UTCDateTime().strftime('%Y%m%d%H%M%S')
    fn_triggers = output_path + my_datestr + '_triggers' + '.csv'
    fn_picks = output_path + my_datestr + '_picks' + '.csv'
    fn_catalog = output_path + my_datestr + '_catalog' + '.csv'
    fn_error = output_path + my_datestr + '_error' + '.txt'
    fn_recent = output_path + my_datestr + '_recents' + '.txt'

    # make folders
    mseedpath = output_path_trig + 'triggers/'
    pngpath = output_path_trig + 'png/'
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(mseedpath):
            os.makedirs(mseedpath)
        if not os.path.exists(pngpath):
            os.makedirs(pngpath)
    except:
        sys.exit('Cannot write to output directory ' + output_path)

    # write trigger file
    header = 'time,duration,SV.PDB01..XN1,SV.PDB02..XN1,SV.PDB03..XN1,SV.PDB04..XN1,SV.PDB05..XN1,SV.PDB06..XN1,SV.PDB07..XN1,SV.PDB08..XN1,SV.PDB09..XN1,SV.PDB10..XN1,SV.PDB11..XN1,SV.PDB12..XN1,SV.OT01..XN1,SV.OT02..XN1,SV.OT03..XN1,SV.OT04..XN1,SV.OT05..XN1,SV.OT06..XN1,SV.OT07..XN1,SV.OT08..XN1,SV.OT09..XN1,SV.OT10..XN1,SV.OT11..XN1,SV.OT12..XN1,SV.PDT1..XNZ,SV.PDT1..XNX,SV.PDT1..XNY,SV.PDB3..XNZ,SV.PDB3..XNX,SV.PDB3..XNY,SV.PDB4..XNZ,SV.PDB4..XNX,SV.PDB4..XNY,SV.PDB6..XNZ,SV.PDB6..XNX,SV.PDB6..XNY,SV.PSB7..XNZ,SV.PSB7..XNX,SV.PSB7..XNY,SV.PSB9..XNZ,SV.PSB9..XNX,SV.PSB9..XNY,SV.PST10..XNZ,SV.PST10..XNX,SV.PST10..XNY,SV.PST12..XNZ,SV.PST12..XNX,SV.PST12..XNY,SV.OB13..XNZ,SV.OB13..XNX,SV.OB13..XNY,SV.OB15..XNZ,SV.OB15..XNX,SV.OB15..XNY,SV.OT16..XNZ,SV.OT16..XNX,SV.OT16..XNY,SV.OT18..XNZ,SV.OT18..XNX,SV.OT18..XNY,SV.CMon..,SV.CTrig..,SV.CEnc..,filename'
    f_triggers = open(fn_triggers, 'wb')
    f_triggers.write(header + '\n')
    f_triggers.close()

    f_recent = open(fn_recent, 'wb')
    f_recent.close()

    stations = ['PDB01', 'PDB02', 'PDB03', 'PDB04', 'PDB05', 'PDB06', 'PDB07', 'PDB08', 'PDB09', 'PDB10', 'PDB11', 'PDB12',
                'OT01', 'OT02', 'OT03', 'OT04', 'OT05', 'OT06', 'OT07', 'OT08', 'OT09', 'OT10', 'OT11', 'OT12',
                'PDT1', 'PDB3', 'PDB4', 'PDB6', 'PSB7', 'PSB9', 'PST10', 'PST12', 'OB13', 'OB15', 'OT16', 'OT18']
    hydrophones = ['PDB01', 'PDB02', 'PDB03', 'PDB04', 'PDB05', 'PDB06', 'PDB07', 'PDB08', 'PDB09', 'PDB10', 'PDB11', 'PDB12',
                'OT01', 'OT02', 'OT03', 'OT04', 'OT05', 'OT06', 'OT07', 'OT08', 'OT09', 'OT10', 'OT11', 'OT12']
    accelerometers = ['PDT1', 'PDB3', 'PDB4', 'PDB6', 'PSB7', 'PSB9', 'PST10', 'PST12', 'OB13', 'OB15', 'OT16', 'OT18']

    pd.options.mode.chained_assignment = None  # default='warn'

    setup_logging(options)
    sys.exit(main(options))
