# python application

All the customization options available in the *c++* version will also be available in *python*. For a basic example the file `cart-pole.py` is extensively commented.

* The script `launchSim.sh` is what smarties actually launches. It can contain additional runtime parameters but the first argument should be the socket id given by smarties at runtime.

* Create an executable script `setup.sh`. The script must:
    - Place in `${BASEPATH}${RUNFOLDER}/`  the `launchSim.sh` script that smarties needs to run to launch your application.  As default, smarties will launch the script `launchSim.sh`, if your script has a different name, or you want to launch an executable, edit the settings variable `launchfile`. Be careful not to name the launch script `launch_smarties.sh` or `run.sh` as that is overwritten by smarties' launch script
    - Place in `${BASEPATH}${RUNFOLDER}/` the python script.
    - Place in `${BASEPATH}${RUNFOLDER}/` the base Communicator class such that it can be imported by your python code.
