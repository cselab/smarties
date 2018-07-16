function run_cases {
COMMNAME=Racer

BUFFSIZE="524288"
IMPSAMPR="4"

for RUNTRIAL in "1" "2" "3" "4" "5"; do

POSTFIX=${COMMNAME}_R${IMPSAMPR}_N${BUFFSIZE}_D${TOLPARAM}_B${BATCHNUM}_O${EPERSTEP}_L${NETLRATE}_S${LAYRSIZE}_TRIAL${RUNTRIAL}
NMASTERS=1
echo $POSTFIX

source launchDaint_openai.sh humanw_${POSTFIX} Humanoid-v2    settings_bench_args_hyper.sh
source launchDaint_openai.sh spider_${POSTFIX} Ant-v2         settings_bench_args_hyper.sh
source launchDaint_openai.sh walker_${POSTFIX} Walker2d-v2    settings_bench_args_hyper.sh
source launchDaint_openai.sh cheeta_${POSTFIX} HalfCheetah-v2 settings_bench_args_hyper.sh

sleep 5

#source launchDaint_openai.sh standu_${POSTFIX} HumanoidStandup-v2        settings_bench_args_hyper.sh
#source launchDaint_openai.sh invpnd_${POSTFIX} InvertedPendulum-v2       settings_bench_args_hyper.sh
#source launchDaint_openai.sh dblpnd_${POSTFIX} InvertedDoublePendulum-v2 settings_bench_args_hyper.sh
#source launchDaint_openai.sh swimmr_${POSTFIX} Swimmer-v2                settings_bench_args_hyper.sh
#source launchDaint_openai.sh hopper_${POSTFIX} Hopper-v2                 settings_bench_args_hyper.sh
#source launchDaint_openai.sh reachr_${POSTFIX} Reacher-v2                settings_bench_args_hyper.sh

done #RUNTRIAL

}

EXPTYPE="GAUS"
NEXPERTS="1"

make -C ../../makefiles clean
make -C ../../makefiles config=fit -j exp=$EXPTYPE nexp=$NEXPERTS


NETLRATE="0.0001"
LAYRSIZE="128"
TOLPARAM="0.1"
EPERSTEP="1.0"
BATCHNUM="256"

for BATCHNUM in "64" "128" "512"; do
run_cases
done
BATCHNUM="256" #reset

for EPERSTEP in "0.2" "0.5" "2.0" "4.0"; do
run_cases
done
EPERSTEP="1.0" #reset

for TOLPARAM in "0.05" "0.15" "0.20"; do
run_cases
done
TOLPARAM="0.1" #reset

for NETLRATE in "0.0003" "0.00003"; do
run_cases
done
NETLRATE="0.0001" #reset

#for LAYRSIZE in "92" "180"; do
#run_cases
#done
#LAYRSIZE="128" #reset

