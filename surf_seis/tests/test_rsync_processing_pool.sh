# Go to SURF_SEIS
cd ~/SURF_SEIS || exit

## I recommend that you start with a clean test folder,
## but I wouldn't want someone to run this accidentally on something else.
rm -rf surf_seis/tests/tmp

mkdir -p surf_seis/tests/tmp/rsync_to_here

#Fire up the autocruncher in the background:
python -m surf_seis.realtime_vibbox_proc surf_seis/tests/test_surf_seisrc &
AUTOCRUNCHER_PID=$!
#Wait for it to boot
sleep 1

echo "Starting rync of two test vibbox files"
rsync surf_seis/tests/test_data/vbox_201805251520453421.dat surf_seis/tests/tmp/rsync_to_here

#Wait till all the processing has completed, then close down the autocruncher:
echo "Waiting for all processes to complete..."
sleep 360
kill $AUTOCRUNCHER_PID
echo "Process has been killed, why not check the logs?"
