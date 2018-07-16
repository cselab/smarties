cp ../apps/test_two_cart_poles/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/test_two_cart_poles/cart-pole ${BASEPATH}${RUNFOLDER}/

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --bSharedPol 0 "
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh
