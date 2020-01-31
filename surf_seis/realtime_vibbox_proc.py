#!/usr/bin/env python

"""
A working (and used!) example of using pyinotify to trigger data reduction.

A multiprocessing pool is used to perform the reduction in
an asynchronous and parallel fashion.
"""

import os
import sys
import uuid
import obspy
import shutil
import logging
import logging.handlers
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from io import StringIO
from configparser import ConfigParser

import pyinotify
from surf_seis.phasepapy.phasepicker import aicdpicker
from pyproj import Proj, transform

from surf_seis.vibbox import (vibbox_read, vibbox_preprocess, vibbox_trigger,
                              vibbox_checktriggers, vibbox_custom_filter
                              )

logger = logging.getLogger()


class RsyncNewFileHandler(pyinotify.ProcessEvent):
    """Identifies new rsync'ed files and passes their path for processing.

    rsync creates temporary files with a `.` prefix and random 6 letter suffix,
    then renames these to the original filename when the transfer is complete.
    To reliably catch (only) new transfers while coping with this
    file-shuffling, we must do a little bit of tedious file tracking, using
    the internal dict `tempfiles`.

    Note we only track those files satisfying the condition
    ``file_predicate(basename)==True``.
    """
    def my_init(self, file_predicate, file_processor):
        self.mask = pyinotify.IN_MOVED_TO | pyinotify.IN_CLOSE_WRITE
        self.predicate = file_predicate
        self.process = file_processor

    def process_IN_MOVED_TO(self, event):
        # Now rsync has renamed the file to drop the temporary suffix.
        # NB event.name == basename(event.pathname) AFAICT
        logger.info('IN_MOVED_TO Sending for processing: %s', event.pathname)
        self.process(event.pathname)

    def process_IN_CLOSE_WRITE(self, event):
        # Now rsync has renamed the file to drop the temporary suffix.
        # NB event.name == basename(event.pathname) AFAICT
        logger.info('IN_CLOSE_WRITE Sending for processing: %s', event.pathname)
        self.process(event.pathname)


class SURF_SEIS():
    def __init__(self, config):
        # Setup directories
        self.watch_dir = config.get('paths', 'watch_dir').strip()
        self.log_dir = config.get('paths', 'log_dir').strip()
        self.output_dir = config.get('paths', 'output_dir').strip()
        self.trigger_dir = config.get('paths', 'trigger_dir').strip()
        self.external_dir = config.get('paths', 'external_dir').strip()
        self.hypoinv_path = config.get('plugins', 'hypinv_path').strip()
        self.tmp_path = os.path.join(self.output_dir, str(uuid.uuid4()))
        self.network = config.get('instruments', 'network').strip()
        self.nthreads = int(config.get('misc', 'nthreads').strip())
        self.stations = [s.strip() for s in
                         config.get('instruments', 'stations').split(',')]
        self.hydrophones = [s.strip() for s in
                            config.get('instruments', 'stations').split(',')]
        self.accelerometers = [s.strip() for s in
                               config.get('instruments', 'stations').split(',')]
        # Build channels array for vibbox
        self.channels = []
        for sta in self.stations:
            if sta in self.hydrophones:
                self.channels.append('{}.{}..XN1'.format(self.network, sta))
            elif sta in self.accelerometers:
                self.channels.extend(['{}.{}..{}'.format(self.network,
                                                         sta, chan)
                                      for chan in ['XNZ', 'XNX', 'XNY']])
        # Output filenames
        my_datestr = obspy.UTCDateTime().strftime('%Y%m%d%H%M%S')
        out_root = os.path.join(self.output_dir, my_datestr)
        trig_root = os.path.join(self.trigger_dir, my_datestr)
        self.fn_triggers = '{}_triggers.csv'.format(trig_root)
        self.fn_picks = '{}_picks.csv'.format(out_root)
        self.fn_catalog = '{}_catalog.csv'.format(out_root)
        self.fn_error = '{}_error.txt'.format(out_root)
        self.fn_recent = '{}_recents.txt'.format(out_root)
        # Build directory tree if needed
        self.mseedpath = os.path.join(self.trigger_dir, 'triggers')
        self.pngpath = os.path.join(self.trigger_dir, 'png')
        for pth in [self.output_dir, self.mseedpath, self.pngpath]:
            if not os.path.exists(pth):
                os.makedirs(pth)
        # Start trigger/recents file
        chan_string = ','.join(self.channels)
        header = '{},{},{},{}.CMon..,{}.CTrig..,{}.CEnc..,filename\n'.format(
            'time', 'duration', chan_string, self.network, self.network,
            self.network
        )
        with open(self.fn_triggers, 'w') as f_triggers:
            f_triggers.write(header)
        with open(self.fn_recent, 'w') as f_recent:
            f_recent.close()
        # Copy over the hypoinverse files to the tmp directory
        try:
            shutil.copytree(self.hypoinv_path, self.tmp_path)
        except Exception as e:
            logger.info(e)
            logger.info('Cannot write {} to tmp directory'.format(
                self.hypoinv_path, self.tmp_path))

    def process_file_trigger(self, fname):
        try:
            st = vibbox_read(fname)
            logger.info('Read {}'.format(fname))
        except Exception as e:
            logger.info(e)
            logger.info('Cannot read {}'.format(fname))
        try:
            st = vibbox_preprocess(st)
        except Exception as e:
            logger.debug(e)
            logger.debug('Error in preprocessing {}'.format(fname))
        try:
            new_triggers = vibbox_trigger(st.copy(), num=20)
            # clean triggers
            new_triggers = vibbox_checktriggers(new_triggers, st.copy())
            res = new_triggers.to_csv(header=False, float_format='%5.4f',
                                      index=False)
        except Exception as e:
            logger.debug(e)
            logger.debug('Error in triggering {}'.format(fname))
            print('Returning zero here')
            return 0
        try:
            st = vibbox_custom_filter(st)
        except Exception as e:
            logger.debug(e)
            logger.debug('Error in filtering {}'.format(fname))
        try:
            for index, ev in new_triggers.iterrows():
                ste = st.copy().trim(starttime=ev['time'] - 0.01,
                                     endtime=ev['time'] + ev['duration'] + 0.01)
                outname = os.path.join(
                    self.mseedpath,
                    '{:10.2f}.mseed'.format(ev['time'].timestamp)
                )
                ste.write(outname, format='mseed')
        except Exception as e:
            logger.info(e)
            logger.info('Error in saving {}'.format(fname))
            return 0
        return new_triggers


    def process_file_pick(self, trigger):
        # Calculate trigger strength on Accelerometers and dont pick if small
        trigger_strength = np.prod(trigger[26:62])
        if trigger_strength < 10000:
            return 0
        # check if we trigger at a minimum of 4 accelerometers
        trig_at_station = np.zeros(len(self.stations))
        triggering_stations = []
        for ii in np.arange(2, 62):
            my_station = trigger.index[ii][3:-5]
            if trigger[ii] > 1.3:
                trig_at_station[self.stations.index(my_station)] = 1
                if not my_station in triggering_stations:
                    triggering_stations.append(my_station)
        if np.sum(trig_at_station[24:]) <= 4:
            return 0
        fname = os.path.join(
            self.mseedpath, '{:10.2f}.mseed'.format(trigger['time'].timestamp)
        )
        try:
            logger.info('Reading {}'.format(fname))
            ste = obspy.read(fname)
        except Exception as e:
            logger.info(e)
            logger.info('Error while reading {}'.format(fname))
            return 0
        try:
            starttime = ste[0].stats.starttime
            endtime = ste[0].stats.endtime
        except Exception as e:
            logger.debug(e)
            logger.debug('Cannot determine start or endtime {}'.format(fname))
            return 0
        try:
            ste = ste.trim(starttime=starttime, endtime=endtime - 0.01)
        except Exception as e:
            logger.debug(e)
            logger.debug('Error in trimming {}'.format(fname))
            return 0
        try:
            picks = self.pick_event(ste, triggering_stations, trigger['time'])
            if len(picks) > 0:
                logger.debug('{} picked'.format(fname))
                logger.debug(picks)
            else:
                logger.debug('No picks in {}'.format(fname))
                return 0
        except Exception as e:
            logger.debug(e)
            logger.debug('Error in picking {}'.format(fname))
            return 0
        try:
            if len(picks) > 300:
                self.plot_picks(picks, ste)
        except Exception as e:
            logger.debug(e)
            logger.debug('Error in plotting ' + fname)
            return 0
        return picks


    def plot_picks(self, my_picks, st):
        """
        Plot up the picks
        :param my_picks:
        :param st: Stream to plot
        :return:
        """
        colspecs = [(0, 14), (14, 42), (42, 51), (51, 79), (79, 80), (80, 87)]
        names = ['eventid', 'origin', 'station', 'pick', 'phase', 'snr']
        my_picks = pd.read_fwf(StringIO(my_picks), colspecs=colspecs, names=names)
        my_picks['origin'] = my_picks['origin'].apply(obspy.UTCDateTime)
        my_picks['pick'] = my_picks['pick'].apply(obspy.UTCDateTime)

        num_picks= len(my_picks)
        starttime = st[0].stats.starttime
        dt = st[0].stats.delta
        fig, axs = plt.subplots(num_picks + 1, 1, sharex=True)
        plt.suptitle('event {}'.format(starttime), color='red')
        for ii in range(num_picks):
            station = my_picks['station'].iloc[ii]
            tr = st.select(station=station)
            axs[ii].plot(tr[0].data, c='k')
            axs[ii].set_yticks([])
            ix = (my_picks['pick'].iloc[ii] - starttime) / dt
            axs[ii].axvline(x=ix, c=(1.0, 0.0, 0.0))
            ylim = axs[ii].get_ylim()
            axs[ii].text(0, (ylim[1] - ylim[0])* 0.9 + ylim[0], station)
        # plot CASSM signal
        axs[-1].plot(st[60].data, c='k', label='CASSM')
        axs[-1].set_yticks([])
        axs[-1].legend()
        fig.set_size_inches(10.5, 2 * (num_picks + 1))
        plt.tight_layout(pad=0.0, h_pad=0.0)
        plt.savefig(os.path.join(
            self.pngpath, '{:10.2f}.png'.format(my_picks['eventid'].iloc[0]))
        )
        plt.close()
        plt.gcf().clear()
        plt.cla()
        plt.clf()


    def pick_event(self, ste, stations, time):
        Picker = aicdpicker.AICDPicker(
            t_ma=0.001, nsigma=8, t_up=0.01, nr_len=0.002, nr_coeff=2,
            pol_len=100, pol_coeff=5, uncert_coeff=3
        )
        output = StringIO()
        for sta in stations:
            if sta in self.hydrophones: # hydrophones
                station = sta.split('.')[0]
                tr = ste.select(station=station)[0]
                scnl, picks, polarity, snr, uncert = Picker.picks(tr)
                if len(picks) > 0:
                    pickstring = '{:10.2f} {} {:8s} {} P {:5.0f}\n'.format(
                        time.timestamp, time, sta, picks[0], snr[0]
                    )
                    output.write(pickstring)
            elif sta in self.accelerometers: # accelerometers
                got_pick = False
                station = sta.split('.')[0]
                tr = ste.select(station = station)
                picks = np.zeros(3)
                snr = np.zeros(3)
                for ii in range(3):
                    scnl, t_picks, polarity, t_snr, uncert = Picker.picks(tr[ii])
                    if len(t_picks) > 0:
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
                    pickstring = '{:10.2f} {} {:8s} {} P {:5.0f}\n'.format(
                        time.timestamp, time, sta, obspy.UTCDateTime(picks), snr
                    )
                    output.write(pickstring)
        return output.getvalue()


    def generate_hyp_input(self, df_picks):
        # output for hypoinverse
        fn_hypinput = os.path.join(self.tmp_path, 'hyp_input.dat')
        # set to 1 for new catalog or larger to continue catalog
        eventid_0 = 1
        min_picks = 5
        only_accelerometers = False
        df_picks['origin'] = df_picks['origin'].apply(obspy.UTCDateTime)
        df_picks['pick'] = df_picks['pick'].apply(obspy.UTCDateTime)
        df_picks = df_picks[df_picks['station'] != 'CTrig']
        if only_accelerometers:
            df_picks = df_picks[df_picks['station'].str.contains('A')]
        eventid = np.unique(df_picks['eventid'])
        df_picks['eventid'] = df_picks['eventid'].replace(
            to_replace=eventid, value=np.arange(len(eventid)) + eventid_0
        )
        df_picks = df_picks.drop_duplicates(subset=['eventid', 'station'],
                                            keep='first')
        # need to log triggers to file
        logger.debug(str(len(eventid)) + ' events will be written')
        df_stations = pd.read_csv(
            os.path.join(self.hypoinv_path, 'stations_local_coordinates.txt'),
            delimiter='\t'
        )
        df_stations['station'] = df_stations['station'].str.rstrip()
        df_stations['z'] = -df_stations['z']
        mean_x0 = df_stations['x'].mean()
        mean_y0 = df_stations['y'].mean()
        max_z0 = df_stations['z'].max()
        # TODO These need to go in a config file
        proj_WGS84 = Proj(init='epsg:4326') # WGS-84
        proj_utm13N = Proj(init='epsg:32613') # UTM 13N
        easting_0 = 598445
        northing_0 = 4912167
        depth_0 = 0
        # scale by a factor of 1000 (and from km to m)
        df_stations['easting_scaled'] = (df_stations['x'] -
                                         mean_x0) * 1e6 + easting_0
        df_stations['northing_scaled'] = (df_stations['y'] -
                                          mean_y0) * 1e6 + northing_0
        # scale by a factor of 1000 (stay with km units)
        # station elevations with positive is up
        df_stations['depth_scaled'] = (df_stations['z'] -
                                       max_z0) * 1e3 + depth_0
        df_stations['lon_scaled'], df_stations['lat_scaled'] = transform(
            proj_utm13N, proj_WGS84, df_stations['easting_scaled'].values,
            df_stations['northing_scaled'].values
        )
        df_stations['lon_deg'] = np.ceil(df_stations['lon_scaled'])
        df_stations['lon_min'] = abs(df_stations['lon_scaled'] -
                                     df_stations['lon_deg']) * 60
        df_stations['lat_deg'] = np.floor(df_stations['lat_scaled'])
        df_stations['lat_min'] = abs(df_stations['lat_scaled'] -
                                     df_stations['lat_deg']) * 60
        events = df_picks['eventid'].unique()
        phasefile = open(fn_hypinput, 'w')
        try_to_locate = False
        for ev in events:
            picks = df_picks.loc[df_picks['eventid'] == ev]
            num_picks = len(picks)
            if num_picks <= min_picks:
                continue
            try_to_locate = True
            origin_time = picks['pick'].min()
            origin_station = picks['station'].loc[picks['pick'] == origin_time]
            origin_station = df_stations[df_stations['station'] ==
                                         origin_station.values[0]]
            phasefile.write(
                '{}{:d}{:2.0f} {:4.0f}{:3d} {:4.0f} {:4.0f}{:3.0f}{:107d}\n'.format(
                    origin_time.strftime('%Y%m%d%H%M%S'),
                    int(origin_time.microsecond / 1e4),
                    float(origin_station.lat_scaled),
                    float(origin_station.lat_scaled -
                          int(origin_station.lat_scaled)) * 6000.,
                    int(abs(origin_station.lon_scaled)),
                    abs(float(origin_station.lon_scaled -
                              int(origin_station.lon_scaled)) * 6000.),
                    float(origin_station.depth_scaled) * -100., 0., np.int(ev)
                    )
            )
            # phasefile.write(
                        # '{:4d}'.format(origin_time.year) +
                        # '{:02d}'.format(origin_time.month) +
                        # '{:02d}'.format(origin_time.day) +
                        # '{:02d}'.format(origin_time.hour) +
                        # '{:02d}'.format(origin_time.minute) +
                        # '{:02d}'.format(origin_time.second) +
                        # '{:02d}'.format(int(origin_time.microsecond / 1e4))  +
                        # '{:2.0f}'.format(float(origin_station.lat_scaled)) + ' ' +
                        # '{:4.0f}'.format(float(origin_station.lat_scaled - int(origin_station.lat_scaled)) * 6000) +
                        # '{:3d}'.format(int(abs(origin_station.lon_scaled))) + ' ' +
                        # '{:4.0f}'.format(abs(float(origin_station.lon_scaled - int(origin_station.lon_scaled)) * 6000)) + ' ' +
                        # '{:4.0f}'.format(float(origin_station.depth_scaled) * -100) +
                        # '{:3.0f}'.format(float(0)) +
                        # '{:107d}'.format(np.int(ev)) + '\n')
            # phase lines
            for index, pick in picks.iterrows():
                traveltime_scaled = (pick['pick'] - origin_time) * 1000
                picktime_scaled = pick['pick'] + traveltime_scaled
                if pick['phase'] == 'P':
                    phasefile.write(
                        '{:5s}{:<9}P 0{:04d}{:02d}{:02d}{:02d}{:02d}{:5.0f}{:>7}{:>5}{:>4}\n'.format(
                            pick.station, self.network, picktime_scaled.year,
                            picktime_scaled.month, picktime_scaled.day,
                            picktime_scaled.hour, picktime_scaled.minute,
                            (picktime_scaled.second +
                             picktime_scaled.microsecond / 1e6) * 100,
                            '101', '0', '0'
                        )
                    )
                    # phasefile.write('{:5s}'.format(pick.station) + 'SV       P 0' +
                    #                 '{:04d}'.format(picktime_scaled.year) + '{:02d}'.format(picktime_scaled.month) + '{:02d}'.format(picktime_scaled.day) +
                    #                  '{:02d}'.format(picktime_scaled.hour) + '{:02d}'.format(picktime_scaled.minute) +
                    #                  '{:5.0f}'.format((picktime_scaled.second + picktime_scaled.microsecond/1e6) * 100) + '   0101    0   0' + '\n')
                if pick['phase'] == 'S':
                    phasefile.write(
                        '{:5s}{:<9}  4{:04d}{:02d}{:02d}{:02d}{:02d}   0   0  0 {:5.0f} S 2\n'.format(
                            pick.station, picktime_scaled.year, self.network,
                            picktime_scaled.month, picktime_scaled.day,
                            picktime_scaled.hour, picktime_scaled.minute,
                            (picktime_scaled.second +
                             picktime_scaled.microsecond / 1e6) * 100,
                            '101', '0', '0'
                        )
                    )
                    # phasefile.write('{:5s}'.format(pick.station_compact) + 'SV         4' +
                    #                 '{:04d}'.format(picktime_scaled.year) + '{:02d}'.format(picktime_scaled.month) + '{:02d}'.format(picktime_scaled.day) +
                    #                  '{:02d}'.format(picktime_scaled.hour) + '{:02d}'.format(picktime_scaled.minute) +
                    #                  '   0   0  0 ' + '{:5.0f}'.format((picktime_scaled.second + picktime_scaled.microsecond/1e6) * 100) + ' S 2\n')
            phasefile.write('{:>64}{:6d}\n'.format('', np.int(ev)))
        phasefile.close()
        return try_to_locate


    def run_hypoinverse(self):
        from subprocess import call
        root_dir = os.getcwd()
        os.chdir(self.tmp_path)
        call('./hyp1.40_ms')
        os.chdir(root_dir)


    def postprocess_hyp(self):
        infile = os.path.join(self.tmp_path, 'hyp_output.csv')
        colspecs = [(0, 24), (24, 31), (32, 42), (43, 48),
                    (48, 62), (62, 64), (65, 68), (69, 74),
                    (75, 81), (82, 87), (88, 92), (93, 95), (97, 108)]
        names = ['date_scaled', 'lat_scaled', 'lon_scaled',
                 'depth_scaled', 'mag', 'num', 'gap', 'dmin',
                 'rms', 'erh', 'erz', 'qasr', 'eventid']
        df_loc = pd.read_fwf(infile, colspecs=colspecs, names=names, skiprows=1)
        # check if any events have been located
        if len(df_loc)==0:
            return 0
        df_loc['date_scaled'] = df_loc['date_scaled'].apply(obspy.UTCDateTime)
        df_loc['date'] = 0
        # TODO Work to be done to gernalize this...
        # TODO Require some info from user in config, others (like staiton...
        # TODO ...locations) in separate files with hard-coded names?
        df_stations = pd.read_csv(
            os.path.join(self.hypoinv_path, 'stations_local_coordinates.txt'),
            delimiter='\t'
        )
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
        df_loc['easting_scaled'], df_loc['northing_scaled'] = transform(
            proj_WGS84, proj_utm13N, df_loc['lon_scaled'].values,
            df_loc['lat_scaled'].values
        )
        df_loc['x'] = ((df_loc['easting_scaled'] - easting_0) /
                       1e6 + mean_x0) * 1000
        df_loc['y'] = ((df_loc['northing_scaled'] - northing_0) /
                       1e6 + mean_y0) * 1000
        df_loc['z'] = ((df_loc['depth_scaled'] - depth_0) /
                       1000 - max_z0) * 1000
        df_picks = pd.read_csv(os.path.join(self.tmp_path, 'picks.csv'))
        df_picks['origin'] = df_picks['origin'].apply(obspy.UTCDateTime)
        df_picks['pick'] = df_picks['pick'].apply(obspy.UTCDateTime)
        df_picks = df_picks[df_picks['station'] != 'CTrig']
        eventid = np.unique(df_picks['eventid'])
        df_picks['eventid'] = df_picks['eventid'].replace(
            to_replace=eventid, value=np.arange(len(eventid)) + 1
        )
        # rescale origin time
        for index, ev in df_loc.iterrows():
            picks = df_picks[df_picks['eventid'] == ev.eventid]
            min_pick = picks['pick'].min()
            date_scaled = df_loc['date_scaled'][df_loc['eventid'] ==
                                                ev.eventid].tolist()[0]
            traveltime = (min_pick - date_scaled)
            df_loc['date'][df_loc['eventid'] == ev.eventid] = (
                    date_scaled + traveltime - traveltime / 1000.0
            )
        return df_loc


    def process_rawfile(self, file):
        if not self.is_rawfile(file):
            return 'Finished processing file, not a raw file: {}'.format(file)
        try:
            with open(self.fn_recent) as f:
                 recent_files = f.read().splitlines()
            if file in recent_files:
                 logger.debug('File was already processed: {}'.format(file))
                 return 'File {} already processed'.format(file)
            else:
                 logger.debug('File not yet processed send '.format(file))
                 recent_files.append(file)
                 if len(recent_files) > 100:
                     recent_files = recent_files[1:]
                 with open(self.fn_recent, 'w') as f:
                     for item in recent_files:
                         f.write('{}\n'.format(item))
        except Exception as e:
            logger.debug(e)
            logger.debug('Cannot determine if in recent files '.format(file))
        df_triggers = self.process_file_trigger(file)
        filename = os.path.split(file)[-1]
        try:
            # this may cause issues with file systems, use subprocess.call
            # instead shutil.move(file, os.path.join(options.ext_dir, filename))
            subprocess.call(['mv', file,
                             os.path.join(self.external_dir, filename)])
            logger.info('Moved to external drive {}'.format(
                os.path.join(self.external_dir, filename)))
        except Exception as e:
            logger.info(e)
            logger.info('Cannot move to external drive '.format(
                os.path.join(self.external_dir, filename)))
        try:
            if len(df_triggers) > 0:
                df_triggers.to_csv(self.fn_triggers, mode='a', header=False,
                                   index=False)
        except Exception as e:
            logger.info(e)
            logger.info('Cannot write triggers')
        try:
            colspecs = [(0, 14), (14, 42), (42, 51),
                        (51, 79), (79, 80), (80, 87)]
            names = ['eventid', 'origin', 'station', 'pick', 'phase', 'snr']
            df_picks = pd.DataFrame()
            for index, trig in df_triggers.iterrows():
                new_picks = self.process_file_pick(trig)
                if new_picks != 0:
                    df_new_picks = pd.read_fwf(StringIO(new_picks),
                                               colspecs=colspecs, names=names)
                    df_picks = df_picks.append(df_new_picks)
            df_events = pd.DataFrame()
            if len(df_picks) > 0:
                print('In writing loop')
                print(os.getcwd())
                # One for hypoinverse, one for posterity
                print(os.path.abspath(os.path.join(self.tmp_path, 'picks.csv')))
                df_picks.to_csv(os.path.abspath(os.path.join(self.tmp_path,
                                                             'picks.csv')),
                                index=False,
                                mode='w')
                df_picks.to_csv(os.path.abspath(self.fn_picks), header=False,
                                index=False, mode='w')
        except Exception as e:
            logger.info(e)
            logger.info('Error in processing picks')
        try:
            if len(df_picks) > 0:
                # Compile scaled Hypoinverse input
                try_to_locate = self.generate_hyp_input(df_picks)
                # Run Hypoinverse
                if try_to_locate:
                    self.run_hypoinverse()
                    # Postprocess hypoinverse
                    df_events = self.postprocess_hyp()
                    df_events.to_csv(self.fn_catalog, header=False,
                                     mode='a', index=False)
        except Exception as e:
            logger.info(e)
            logger.info('Error during locating events')
        # cleanup
        # try:
        #     if os.path.exists(self.tmp_path):
        #         shutil.rmtree(self.tmp_path)
        # except:
        #     sys.exit('Cannot clean temp directory {}'.format(self.tmp_path))
        return 'Finished processing file {}.\n{} triggers found\n{} events located'.format(
            file, len(df_triggers), len(df_events))


    def is_rawfile(self, filename):
        """Predicate function for identifying incoming AMI data"""
        if filename.endswith('.dat'):
            return True
        return False


    def newfile_callback(self, file):
        """Used to log the 'job complete' / error message in the master thread."""
        logger.info('*** File submitted for processing: {}'.format(file))


    def processed_callback(self, summary):
        """Used to log the 'job complete' / error message in the master thread."""
        logger.info('*** Job complete: {}'.format(summary))


    def log_preamble(self):
        logger.info("***********")
        logger.info('Watching {}'.format(self.watch_dir))
        logger.info('Output dir {}'.format(self.output_dir))
        logger.info('Log dir {}'.format(self.log_dir))
        logger.info('External dir {}'.format(self.external_dir))
        logger.info("***********")


    def setup_logging(self):
        """Set up basic (INFO level) and debug logfiles

        These should list successful reductions, and any errors encountered.
        We also copy the basic log to STDOUT, but it is expected that
        the monitor script will be daemonised / run in a screen in the background.
        """
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        log_filename = os.path.join(self.log_dir, 'autocruncher_log')
        date_fmt = "%a %d %H:%M:%S"
        std_formatter = logging.Formatter(
            '%(asctime)s:%(levelname)s:%(message)s', date_fmt)
        debug_formatter = logging.Formatter(
            '%(asctime)s:%(name)s:%(levelname)s:%(message)s', date_fmt)
        info_logfile = logging.handlers.RotatingFileHandler(
            log_filename, maxBytes=5e7, backupCount=10)
        info_logfile.setFormatter(std_formatter)
        info_logfile.setLevel(logging.INFO)
        debug_logfile = logging.handlers.RotatingFileHandler(
            '{}.debug'.format(log_filename), maxBytes=5e7, backupCount=10)
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

    def run(self):
        """
        Run these things
        :return:
        """
        pool = multiprocessing.Pool(self.nthreads)
        def simply_process_rawfile(file_path):
            """
            Wraps `process_rawfile` to take a single argument (file_path).

            This is the trivial, single threaded version -
            occasionally useful for debugging purposes.
            """
            self.newfile_callback(file_path)
            summary = self.process_rawfile(file_path)
            self.processed_callback(summary)

        def asynchronously_process_rawfile(file_path):
            """
            Wraps `process_rawfile` to take a single argument (file_path).

            This version runs 'process_rawfile' asynchronously via the pool.
            This provides parallel processing, at the cost of being harder to
            debug if anything goes wrong (see notes on exception catching above)
            """

            pool.apply_async(self.process_rawfile, [file_path],
                             callback=self.processed_callback)
        if self.nthreads > 1:
            func = asynchronously_process_rawfile
        else:
            func = simply_process_rawfile
        # Now set the whole thing in motion
        try:
            handler = RsyncNewFileHandler(
                file_predicate=self.is_rawfile,
                file_processor=func
            )
            wm = pyinotify.WatchManager()
            notifier = pyinotify.Notifier(wm, handler)
            wm.add_watch(self.watch_dir, handler.mask, rec=True)
            self.log_preamble()
            notifier.loop()
        finally:
            pool.close()
            pool.join()
        return 0


def main():
    args = sys.argv
    if len(args) == 1:
        msg = 'Must provide a config file: realtime_vibbox_proc [file]'
        raise Exception(msg)
    config_path = args[1]
    config = ConfigParser()
    config.read(config_path)
    surf_seis = SURF_SEIS(config)
    surf_seis.setup_logging()
    sys.exit(surf_seis.run())


if __name__ == '__main__':
    main()
