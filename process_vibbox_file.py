#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:02:17 2018

@author: schoenball

based on
trigger_only_180724.py
pick_only_180731.py
prep_hypoinverse_from_vibbox.py
post_hypoinverse_to_surf.py
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import multiprocessing as mp
import obspy
from StringIO import StringIO
import vibbox_v2 as vibbox
from phasepapy.phasepicker import aicdpicker
import sys
from pyproj import Proj, transform
import uuid
import shutil


def process_file_trigger(fname):
    try:
        st = vibbox.vibbox_read(fname)
        print('Read ' + fname)
    except Exception as e:
        print(e)
        print('Cannot read ' + fname)
    try:
        st = vibbox.vibbox_preprocess(st)
    except Exception as e:
        print(e)
        print('Error in preprocessing ' + fname)
    try:
        new_triggers = vibbox.vibbox_trigger(st.copy())
        # clean triggers
        new_triggers = vibbox.vibbox_checktriggers(new_triggers, st.copy())
        res = new_triggers.to_csv(header=False, float_format='%5.4f', index=False)
    except Exception as e:
        print(e)
        print('Error in triggering ' + fname)
        return 0
    try:
        st = vibbox.vibbox_custom_filter(st)
    except Exception as e:
        print(e)
        print('Error in filtering ' + fname)
    try:
        for index, ev in new_triggers.iterrows():
            ste = st.copy().trim(starttime = ev['time'] - 0.01,  endtime = ev['time'] + ev['duration'] + 0.01)
            outname = mseedpath + '{:10.2f}'.format(ev['time'].timestamp)  + '.mseed'
            ste.write(outname, format='mseed')
    except Exception as e:
        print(e)
        print('Error in saving ' + fname)
        return 0
    if debug_level >= 1:
        print(new_triggers)
    return new_triggers

def process_file_pick(trigger):
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
        print('Reading ' + fname)
        ste = obspy.read(fname)
    except Exception as e:
        print(e)
        print('Error while reading ' + fname)
        return 0
    try:
        starttime = ste[0].stats.starttime
        endtime = ste[0].stats.endtime
    except Exception as e:
        print(e)
        print('Cannot determine start or endtime: ' + fname)
        return 0
    try:
        ste = ste.trim(starttime=starttime, endtime=endtime - 0.01)
    except Exception as e:
        print(e)
        print('Error in trimming ' + fname)
        return 0
    try:
        picks = pick_event(ste, triggering_stations, trigger['time'])
        if len(picks) > 0:
            print(fname + ' picked:')
            print(picks)
        else:
            print('No picks in ' + fname)
            return 0
#        q.put('#' + picks)
    except Exception as e:
        print(e)
        print('Error in picking ' + fname)
        return 0
    try:
        if len(picks) > 300:
            plot_picks(picks, ste)
    except Exception as e:
        print(e)
        print('Error in plotting ' + fname)
        return 0
    return picks

def plot_picks(my_picks, st):
    colspecs = [(0, 14), (14, 42), (42, 51), (51, 79), (79, 80), (80, 87)]
    names = ['eventid', 'origin', 'station', 'pick', 'phase', 'snr']
    my_picks = pd.read_fwf(StringIO(my_picks), colspecs=colspecs, names=names)
    my_picks['origin'] = my_picks['origin'].apply(obspy.UTCDateTime)
    my_picks['pick'] = my_picks['pick'].apply(obspy.UTCDateTime)

    num_picks= len(my_picks)
    if num_picks > 1:
        fig, axs = plt.subplots(num_picks, 1, sharex=True)
        starttime = st[0].stats.starttime
        dt = st[0].stats.delta
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
        fig.set_size_inches(10.5, 2*num_picks)
        plt.tight_layout(pad=0.0, h_pad=0.0)
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
#                print(pickstring)
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
#                print(pickstring)
                output.write(pickstring)
    return output.getvalue()

def generate_hyp_input(df_picks):
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
    print(str(len(eventid)) + ' events will be written')    # need to log triggers to file

    df_stations = pd.read_csv('stations_local_coordinates.txt', delimiter='\t')
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
        shutil.copy('surf.crh', output_path + my_uuid)
        shutil.copy('hypinst', output_path + my_uuid)
        shutil.copy('hyp1.40_ms', output_path + my_uuid)
        shutil.copy('hyp_stations.dat', output_path + my_uuid)
    except:
        sys.exit('Cannot write to output directory ', output_path)
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

def run_hypoinverse():
    from subprocess import call
    os.chdir(output_path + my_uuid)
    call('./hyp1.40_ms')
    os.chdir('../../')
    # clean up
    # shutil.move('hyp_input.dat', output_path + 'hyp_input_' + my_datestr + '.dat')
    # shutil.move('hyp_output.arc', output_path + 'hyp_output_' + my_datestr + '.arc')
    # shutil.move('hyp_output.csv', output_path + 'hyp_output_' + my_datestr + '.csv')
    # shutil.move('hyp_output.prt', output_path + 'hyp_output_' + my_datestr + '.prt')

def postprocess_hyp():
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

    df_stations = pd.read_csv('stations_local_coordinates.txt', delimiter='\t')
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
    df_triggers = process_file_trigger(file)
    if len(df_triggers)>0:
        df_triggers.to_csv(fn_triggers, mode='a', header=False, index=False)
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
    # Compile scaled Hypoinverse input
        try_to_locate = generate_hyp_input(df_picks)
    # Run Hypoinverse
        if try_to_locate:
            run_hypoinverse()
        # Postprocess hypoinverse
            df_events = postprocess_hyp()
            df_events.to_csv(fn_catalog, header=False, mode='a', index=False)
    # cleanup
    try:
        if os.path.exists(output_path + my_uuid):
            shutil.rmtree(output_path + my_uuid)
    except:
        sys.exit('Cannot clean temp directory ', output_path + my_uuid)

    print('Finished processing file ' + file + '.\n' + str(len(df_triggers)) + ' triggers found\n' + str(len(df_events)) + ' events located')

# write out picks
# add results of this file to the global catalog: triggers, picks, post-proc catalog

if __name__ == "__main__":
    n_cpus = 9
    debug_level = 2
    output_path = './out/'

    # output filenames
    my_datestr = obspy.UTCDateTime().strftime('%Y%m%d%H%M%S')
    fn_triggers = output_path + 'triggers_' + my_datestr + '.csv'
    fn_picks = output_path + 'picks_' + my_datestr + '.csv'
    fn_catalog = output_path + 'catalog_' + my_datestr + '.csv'
    fn_error = output_path + 'error_' + my_datestr + '.csv'
    my_uuid = str(uuid.uuid4())

    # make folders
    mseedpath = output_path + 'triggers/'
    pngpath = output_path + 'png/'
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(mseedpath):
            os.makedirs(mseedpath)
        if not os.path.exists(pngpath):
            os.makedirs(pngpath)
        if not os.path.exists(output_path + my_uuid):
            os.makedirs(output_path + my_uuid)
    except:
        sys.exit('Cannot write to output directory ', output_path)

    # write trigger file
    header = 'time,duration,SV.PDB01..XN1,SV.PDB02..XN1,SV.PDB03..XN1,SV.PDB04..XN1,SV.PDB05..XN1,SV.PDB06..XN1,SV.PDB07..XN1,SV.PDB08..XN1,SV.PDB09..XN1,SV.PDB10..XN1,SV.PDB11..XN1,SV.PDB12..XN1,SV.OT01..XN1,SV.OT02..XN1,SV.OT03..XN1,SV.OT04..XN1,SV.OT05..XN1,SV.OT06..XN1,SV.OT07..XN1,SV.OT08..XN1,SV.OT09..XN1,SV.OT10..XN1,SV.OT11..XN1,SV.OT12..XN1,SV.PDT1..XNZ,SV.PDT1..XNX,SV.PDT1..XNY,SV.PDB3..XNZ,SV.PDB3..XNX,SV.PDB3..XNY,SV.PDB4..XNZ,SV.PDB4..XNX,SV.PDB4..XNY,SV.PDB6..XNZ,SV.PDB6..XNX,SV.PDB6..XNY,SV.PSB7..XNZ,SV.PSB7..XNX,SV.PSB7..XNY,SV.PSB9..XNZ,SV.PSB9..XNX,SV.PSB9..XNY,SV.PST10..XNZ,SV.PST10..XNX,SV.PST10..XNY,SV.PST12..XNZ,SV.PST12..XNX,SV.PST12..XNY,SV.OB13..XNZ,SV.OB13..XNX,SV.OB13..XNY,SV.OB15..XNZ,SV.OB15..XNX,SV.OB15..XNY,SV.OT16..XNZ,SV.OT16..XNX,SV.OT16..XNY,SV.OT18..XNZ,SV.OT18..XNX,SV.OT18..XNY,SV.CMon..,SV.CTrig..,SV.CEnc..,filename'
    f_triggers = open(fn_triggers, 'wb')
    f_triggers.write(header + '\n')
    f_triggers.close()

    stations = ['PDB01', 'PDB02', 'PDB03', 'PDB04', 'PDB05', 'PDB06', 'PDB07', 'PDB08', 'PDB09', 'PDB10', 'PDB11', 'PDB12',
                'OT01', 'OT02', 'OT03', 'OT04', 'OT05', 'OT06', 'OT07', 'OT08', 'OT09', 'OT10', 'OT11', 'OT12',
                'PDT1', 'PDB3', 'PDB4', 'PDB6', 'PSB7', 'PSB9', 'PST10', 'PST12', 'OB13', 'OB15', 'OT16', 'OT18']
    hydrophones = ['PDB01', 'PDB02', 'PDB03', 'PDB04', 'PDB05', 'PDB06', 'PDB07', 'PDB08', 'PDB09', 'PDB10', 'PDB11', 'PDB12',
                'OT01', 'OT02', 'OT03', 'OT04', 'OT05', 'OT06', 'OT07', 'OT08', 'OT09', 'OT10', 'OT11', 'OT12']
    accelerometers = ['PDT1', 'PDB3', 'PDB4', 'PDB6', 'PSB7', 'PSB9', 'PST10', 'PST12', 'OB13', 'OB15', 'OT16', 'OT18']

    pd.options.mode.chained_assignment = None  # default='warn'

    #put the following stuff in process_one_file():
#    file = '/home/sigmav/Vibbox_processing/vbox_201807221529567735.dat'
    file = '/home/sigmav/test/vbox_201805222204530380.dat'
    process_rawfile(file)
