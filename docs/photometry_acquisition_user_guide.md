# Photometry user guide
this document describes the individual steps for acquiring photometry data with the FP3002 in the context of the IBL task

- note: this is for DAQ based synchronization

## 1. Activate the IBL environment
on the neurophotometrics computer, open a powershell and run

`C:\iblrigv8\venv\scripts\Activate.ps1`

this will activate the python environment, necessary to recognize any of the other commands

## 2. Start neurophotometry acquisition pipeline
in the shell, run 

`start_neurophotometrics --sync-mode daqami`

or

`start_neurophotometrics -m daqami`

This will start the main script that organizes the entire acquisition, configured for the daq based synchronization scheme. 

It will directly halt and prompt you to start the *daqami* software.

## 3. Start daqami software
Start the daqami software, load in the most recently used configuration.

## 4. Start recording in the daqami software
Hit the red *record* button in the daqami software.

- possible pitfall: make sure to hit the record button and *not* the play button. Hitting the play button will also display the incoming data, but it will not safe anything, and the sessions of that day will not be recoverable.

## 5. Continue photometry pipeline startup
go back to the shell window that prompted you to start the daqami software and hit _enter_

A bonsai workflow window will appear.

## 6. Adjust FP3002 settings
Clicking on the FP3002 bonsai node will give you access to the settings.

adjust camera ROIs, power levels, verify triggers

## 7. Start the neurophotometrics acquisition
Hit the green "play" button in the bonsai workflow to start aquiring data with the neurophotometrics.

Visually confirm that
- light is being emitted from the end of the patch cords
- frame TTLs are being acquired in the daqami software

## 8. Prepare the mouse
Do all the mouse related steps
 - head fixation
 - placing it in the rig
 - 

Prepare IBLrig wizard
- select project, animal, task etc

but _do not start the session yet_ (!)

## 9. Initialize the animal on the server
on the neurophotometrics computer, run 

`start_photometry_task`

with the adequate parameters.

For example, you are running subject `MM-0123` with two fiber implants, one in VTA and one in STR. In the FP3002 software, you have placed roi `G0` over the fiber that connects to VTA, and `G1` to the fiber that connects to the the fiber in STR. You are running the mouse in rig 1, which is connected to sync channel 1. The command will then be

`start_photometry_task --subject MM-0123 --rois G0 G1 --locations VTA STR --sync-channel 1 --sync-mode daqami`

or equivalently

`start_photometry_task -s MM-0123 -r G0 G1 -l VTA STR -c 1 -m daqami`

you can use the `--help` flag to inspect a command for possible options, i.e. `start_photometry_task --help` will not run the command but print out a usage guide.

## 10. start the session in IBLrig
Now start the session in the iblrig wizard

## 11. end of a session
- end the session in IBLrig
- take out the mouse
- for the next mouse, start at step 8.

## 12. end of the day
after all mice have finished their sessions
- click the red stop square in bonsai
- click the red stop square in daqami

