#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:43:55 2018

@author: schoenball

version history:
2018/06/13: first version
2018/11/11: v2: changed channel naming to be in line with miniSEED Guide
2018/12/03: improved the vibbox_checktriggers routine to check if CASSM trigger is contained in stream
            added vibbox_remove_ert_triggers() to remove ERT noise 
2018/12/06: Remove CASSM shorts already in vibbox_trigger; remove it from vibbox_checktriggers
2019/04/16: Corrected the factor scaling the signal by 2**31, propagated change to time signal 
            detection and recognizing CASSM triggers
"""
import os
import numpy as np
import pandas as pd

from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats
from obspy.signal.trigger import coincidence_trigger

def vibbox_preprocess(st):
    # PDB
    ## De-Median PDB hydrophones
    #temp = np.median((st[0].data, st[1].data, st[2].data, st[3].data, st[4].data, st[5].data, 
    #                  st[6].data, st[7].data, st[8].data, st[9].data, st[10].data, st[11].data), axis=0)
    #for ii in np.arange(0,12):
    #    st[ii].data = st[ii].data - temp
    
    # OT
    # align sign OT hydrophones
#    for ii in np.arange(16, 23):
#        st[ii].data = - st[ii].data
    ## De-Median OT hydrophones
    #temp = np.median((st[12].data, st[13].data, st[14].data, st[15].data, st[16].data, st[17].data, 
    #                  st[18].data, st[19].data, st[20].data, st[21].data, st[22].data), axis=0)
    #for ii in np.arange(12,23):
    #    st[ii].data = st[ii].data - temp
    # Hydrophone OT12 is dead
    st[23].data = 0*st[23].data
    
    # Accelerometers
    # PDT_A1
    temp = np.mean((st[24].data, st[25].data, st[26].data), axis=0)
    for ii in np.arange(24,27):
        st[ii].data = st[ii].data - temp
    # PDB_A3
    temp = np.mean((st[27].data, st[28].data, st[29].data), axis=0)
    for ii in np.arange(27,30):
        st[ii].data = st[ii].data - temp
    # PDB_A4
    temp = np.mean((st[30].data, st[31].data, st[32].data), axis=0)
    for ii in np.arange(30,33):
        st[ii].data = st[ii].data - temp
    # PDB_A6
    temp = np.mean((st[33].data, st[34].data, st[35].data), axis=0)
    for ii in np.arange(33,36):
        st[ii].data = st[ii].data - temp
    # PSB_A7
    temp = np.mean((st[36].data, st[37].data, st[38].data), axis=0)
    for ii in np.arange(36,39):
        st[ii].data = st[ii].data - temp
    # PDB_A9
    temp = np.mean((st[39].data, st[40].data, st[41].data), axis=0)
    for ii in np.arange(39,42):
        st[ii].data = st[ii].data - temp
    # PST_A10
    temp = np.mean((st[42].data, st[43].data, st[44].data), axis=0)
    for ii in np.arange(42,45):
        st[ii].data = st[ii].data - temp
    # PST_A12
    temp = np.mean((st[45].data, st[46].data, st[47].data), axis=0)
    for ii in np.arange(45,48):
        st[ii].data = st[ii].data - temp
    # OB_A13
    temp = np.mean((st[48].data, st[49].data, st[50].data), axis=0)
    for ii in np.arange(48,51):
        st[ii].data = st[ii].data - temp
    # OB_A15
    temp = np.mean((st[51].data, st[52].data, st[53].data), axis=0)
    for ii in np.arange(51,54):
        st[ii].data = st[ii].data - temp
    # OT_A16
    temp = np.mean((st[54].data, st[55].data, st[56].data), axis=0)
    for ii in np.arange(54,57):
        st[ii].data = st[ii].data - temp    
    # OT_A18
    temp = np.mean((st[57].data, st[58].data, st[59].data), axis=0)
    for ii in np.arange(57,60):
        st[ii].data = st[ii].data - temp  

    return st

def vibbox_custom_filter(st):
    hydrophones = list(np.arange(0, 24)) # hydrophones +  
    for comp in hydrophones:
        st[comp].filter('bandstop', freqmin=14480, freqmax=14880, corners=2, zerophase=True)
        st[comp].filter('bandstop', freqmin=20500, freqmax=20850, corners=2, zerophase=True)
        st[comp].filter('bandstop', freqmin=21160, freqmax=21560, corners=2, zerophase=True)
        st[comp].filter('bandstop', freqmin=35780, freqmax=36080, corners=2, zerophase=True)
        st[comp].filter('bandstop', freqmin=41200, freqmax=41500, corners=2, zerophase=True)
        st[comp].filter('bandstop', freqmin=42320, freqmax=42920, corners=2, zerophase=True)
    ot_a_16 = list(np.arange(54, 57)) # OT_A.16  
    for comp in ot_a_16:
        st[comp].filter('bandstop', freqmin=42320, freqmax=42920, corners=2, zerophase=True)
    lowpass_42khz = list(np.arange(12, 17)) + list(np.arange(27, 30)) + list(np.arange(45, 48))
    lowpass_42khz.append(49) # OT_H01-04, PDB_A3, PST_A12, OB_A13.X
    for comp in lowpass_42khz:
        st[comp].filter('lowpass', freq=42000, zerophase=True)
    ot_h01_04 = np.arange(12, 17)
    for comp in ot_h01_04:
        st[comp].filter('bandstop', freqmin=33870, freqmax=34470, corners=2, zerophase=True)
        st[comp].filter('bandstop', freqmin=21500, freqmax=24000, corners=2, zerophase=True)        
        
    return(st)


def vibbox_checktriggers(my_triggers, st):
    # declare dead time after each trigger
    deadtime = 0
    for ev_index, ev in my_triggers.iterrows():
        if ev.time > deadtime:
            deadtime = ev.time + ev.duration
        else:
            my_triggers = my_triggers.drop(ev_index)
    ids = list(map(lambda d: d.id, st))
    # remove electrical spikes and ERT cross talk
    for ev_index, ev in my_triggers.iterrows():
        ste = st.copy().trim(starttime=ev['time'] - 0.001,
                             endtime = ev['time'] + ev['duration'] + 0.001)
        # select triggering components
        max_sample = ev[ev > 1.3]
        # find peak during trigger
        for sta_index, sta in max_sample.items():
            if sta_index in ids:
                max_sample[sta_index] = np.argmax(
                    np.abs(ste.select(id=sta_index)[0].data)
                )
            else:
                max_sample = max_sample.drop(sta_index)
        # determine label for trigger
        max_sample = max_sample.sort_values()
        unique_values = np.unique(np.ceil(max_sample.astype(float).values/3.0))
        # real events should have their maxima at any station not closer
        # than 3 samples for more than 20% of triggering traces
        label = (np.float(len(unique_values)) / np.float(len(max_sample))) > 0.8
        if not label:
            my_triggers = my_triggers.drop(ev_index)
    return(my_triggers)


def vibbox_trigger(st, freqmin=1000, freqmax=15000, sta=0.01, lta=0.05, on=1.3,
                   off=1, num=10):
    starttime = st[0].stats.starttime
    st = st[0:63]  # throw out time signal
    cassm = st[61].copy().differentiate()
    st.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    trigs = coincidence_trigger("recstalta", on,  off, st, num,
                                sta=sta, lta=lta
                                )
    st.trigger('recstalta', sta=sta, lta=lta)
    ids = list(map(lambda d: d.id, st))
    columns = ['time', 'duration'] + ids
    df_triggers = pd.DataFrame(columns=columns)
    for trig in trigs:
        # trow out CASSM shots
        if 'SV.CTrig..' in trig['trace_ids']:
            continue
        # spurious triggers at beginning of file, should be done differently
        if trig['duration'] > 0.1:
            continue
        trig_sample = np.int((UTCDateTime(trig['time']) -
                              starttime) * st[0].stats.sampling_rate
                             )
        # check if CASSM trigger fired
        if np.max(cassm.data[(trig_sample - 1000):(1000 + trig_sample +
                                                   np.int(trig['duration'] *
                                                   st[0].stats.sampling_rate))]
                  ) > 10000:
            continue
        current_trigger = {'time': trig['time'], 'duration': trig['duration']}
        for i, tr in enumerate(st):
            if i >= 60:
                continue
            current_trigger[tr.id] = np.max(
                tr.data[trig_sample:trig_sample +
                                    np.int(trig['duration'] *
                                           st[0].stats.sampling_rate)]
            )
        df_triggers = df_triggers.append(current_trigger, ignore_index=True)
    return df_triggers


def vibbox_read(fname, param):
    network = param['General']['stats']['network']
    stations = param['Acquisition']['asdf_settings']['station_naming']
    locations = ['' for i in range(len(stations))]
    channels = param['Acquisition']['asdf_settings']['channel_naming']
    # Find channel PPS (pulse per second)
    try:
        clock_channel = np.where(np.array(stations) == 'PPS')[0][0]
    except IndexError:
        print('No PPS channel in file. Not reading')
        return
    # TODO Everything from here to file open should go in config
    HEADER_SIZE=4
    HEADER_OFFSET=27
    DATA_OFFSET=148
    VOLTAGE_RANGE=10
    with open(fname, "rb") as f:
        f.seek(HEADER_OFFSET, os.SEEK_SET)
        # read header
        H = np.fromfile(f, dtype=np.int32, count=HEADER_SIZE)
        BUFFER_SIZE=H[0]
        FREQUENCY=H[1]
        NUM_OF_BUFFERS=H[2]
        no_channels=H[3]
        # read data
        f.seek(DATA_OFFSET, os.SEEK_SET)
        A = np.fromfile(f, dtype=np.int32,
                        count=BUFFER_SIZE * NUM_OF_BUFFERS)
        A = A.reshape(int(len(A) / no_channels), no_channels)
    import matplotlib.pyplot as plt
    plt.plot(A[:, clock_channel], label='1') # Good here
    # Sanity check on number of channels provided in yaml
    if len(channels) != no_channels:
        print('Number of channels in config file not equal to number in data')
        return
    # TODO What are the following two lines doing?
    A = (2 * VOLTAGE_RANGE * A) - VOLTAGE_RANGE
    plt.plot(A[:, clock_channel], label='2')
    A = A / 4294967296.0
    plt.plot(A[:, clock_channel], label='3')
    plt.legend()
    path, fname = os.path.split(fname)
    try:
        time_to_first_full_second = np.where(A[:, clock_channel] >
                                             (2e7 / 2**31))[0][0] - 3
        print(time_to_first_full_second)
        if time_to_first_full_second > 101000:
            print('Cannot read time signal')
        # in case we start during the time pulse
        if time_to_first_full_second < 0:
            time_to_first_full_second = np.where(A[50000:, clock_channel] >
                                                 (2e7 / 2**31))[0][0] - 3
        print(time_to_first_full_second)
        print(np.int(1e6 * (1 - (np.float(time_to_first_full_second) /
                               FREQUENCY))))
        starttime = UTCDateTime(
            np.int(fname[5:9]), np.int(fname[9:11]), np.int(fname[11:13]),
            np.int(fname[13:15]), np.int(fname[15:17]), np.int(fname[17:19]),
            np.int(1e6 * (1 - (np.float(time_to_first_full_second) /
                               FREQUENCY))))
    except Exception as e:
        print(e)
        print('Cannot read time exact signal: ' + fname +
              '. Taking an approximate one instead')
        starttime = UTCDateTime(
            np.int(fname[5:9]), np.int(fname[9:11]), np.int(fname[11:13]),
            np.int(fname[13:15]), np.int(fname[15:17]), np.int(fname[17:19]),
            np.int(1e2 * np.int(fname[19:23])))
    # arrange it in an obspy stream
    st = Stream()
    for i, sta in enumerate(stations):
        stats = Stats()
        stats.sampling_rate = H[1]
        stats.network = network
        stats.station = sta
        stats.channel = channels[i]
        stats.location = locations[i]
        stats.starttime = starttime
        st.traces.append(Trace(data=A[:, i], header=stats))
    return st, A

