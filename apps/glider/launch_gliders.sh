
for ASPECT in "0.025" "0.050" "0.100" "0.200" "0.400"; do
for DRATIO in "25" "50" "100" "200" "400" "800"; do
for REWARD in "1" "2"; do

make -C ../apps/glider clean
make -C ../apps/glider aspectr=${ASPECT} density=${DRATIO} costfun=${REWARD}

PRRATIO=`printf "%03d" $DRATIO`
echo glider_dt05_alpha${ASPECT}_rho${PRRATIO}_cost${REWARD}
#./launch.sh glider_dacer_dt05_alpha${ASPECT}_rho${PRRATIO}_cost${REWARD} 12 glider settings/settings_DACER_safe.sh
./launch.sh train06_dt05_alpha${ASPECT}_rho${PRRATIO}_cost${REWARD} 12 glider settings/settings_POAC_safe.sh

done
done
done
