
for ASPECT in "0.01" "0.03" "0.1" "0.3" "1"; do
for DRATIO in "10" "50" "200" "500" "1000"; do
for REWARD in "1" "2"; do

make -C ../apps/glider clean
make -C ../apps/glider aspectr=${ASPECT} density=${DRATIO} costfun=${REWARD}

POSTFIX=${RATIO}_R${ASPECT}_C${REWARD}
echo $POSTFIX

./launch.sh glider_dt05_beta_${ASPECT}_rho_${DRATIO}_cost_${REWARD} 24 1 glider settings/settings_POAC.sh none 24

done
done
done
