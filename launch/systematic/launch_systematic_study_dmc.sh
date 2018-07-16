for EXPTYPE in "GAUS"; do
for NEXPERTS in "1"; do

make -C ../../makefiles clean
make -C ../../makefiles config=fit -j exp=$EXPTYPE nexp=$NEXPERTS

for BUFFSIZE in "524288"; do
for IMPSAMPR in "4"; do

for TOLPARAM in "0.1"; do
for BATCHNUM in "256"; do
for EPERSTEP in "1"; do
for RUNTRIAL in "1" "2" "3" "4" "5"; do

POSTFIX=POSTFIX=RACDMC99_R${IMPSAMPR}_N${BUFFSIZE}_D${TOLPARAM}_TRIAL${RUNTRIAL}
NMASTERS=1

declare -a listOfCases=( \
                        "acrobot.swingup_sparse" \
                        "acrobot.swingup" \
                        "ball_in_cup.catch" \
                        "cartpole.swingup" \
                        "cartpole.balance_sparse" \
                        "cartpole.balance" \
                        "cartpole.swingup_sparse" \
                        "cheetah.run" \
                        "finger.spin" \
                        "finger.turn_easy" \
                        "finger.turn_hard" \
                        "fish.upright" \
                        "fish.swim" \
                        "hopper.hop" \
                        "hopper.stand" \
                        "humanoid.run" \
                        "humanoid.walk" \
                        "humanoid.stand" \
                        "manipulator.bring_ball" \
                        "pendulum.swingup" \
                        "point_mass.easy" \
                        "reacher.easy" \
                        "reacher.hard" \
                        "swimmer.swimmer15" \
                        "swimmer.swimmer6" \
                        "walker.run" \
                        "walker.walk" \
                        "walker.stand" \
                      )
#
for RUNTRIAL in "${listOfCases[@]}" ; do

RUNNAME=${RUNTRIAL/./_}
RUNTRIAL=${RUNTRIAL/./ }
echo $RUNTRIAL $RUNNAME
source launchDaint_deepmind.sh ${RUNNAME}_${POSTFIX} $RUNTRIAL settings_bench_args_DMC.sh
sleep 1 #avoid overwhelming server
done

done
done
done
done
done
done

done
done
