# apps

Each folder contains the files required to prepare the run directory for running an application. Multiple folders here might refer to the same environment of smarties.

When calling the launch script (eg. `launch/launch.sh`) the user specifies a folder contained here, from which the script `setup.sh` is executed. The script `setup.sh` as the name suggests must perform any required operation to make it possible to run he environment. Bear in mind that `setup.sh` is called from the directory `../launch`, therefore any relative path should be specified form there.

The base run directory can be found with `${BASEPATH}${RUNFOLDER}/`, however that is the folder where smarties is run. The actual application code will be run in a randomly named `simulation_%06d_%01d` folder. From there smarties launches `../launchSim.sh`, therefore this should be in smarties' base run directory.
