make -C ../apps/test_cpp_cart_pole

cp ../apps/test_cpp_cart_pole/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/test_cpp_cart_pole/cart-pole ${BASEPATH}${RUNFOLDER}/
