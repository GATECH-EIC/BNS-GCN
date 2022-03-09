mkdir results
for N_PARTITIONS in 3 6 10
do
  for SAMPLING_RATE in 0.10 0.01 0.00
  do
    echo -e "\033[1mclean python processes\033[0m"
    sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
    echo -e "\033[1m${N_PARTITIONS} partitions, ${SAMPLING_RATE} sampling rate\033[0m"
    python main.py \
      --dataset yelp \
      --dropout 0.1 \
      --lr 0.001 \
      --n-partitions ${N_PARTITIONS} \
      --n-epochs 3000 \
      --model graphsage \
      --sampling-rate ${SAMPLING_RATE} \
      --n-layers 4 \
      --n-linear 2 \
      --n-hidden 512 \
      --log-every 10 \
      --inductive \
      --use-pp \
      |& tee results/yelp_n${N_PARTITIONS}_p${SAMPLING_RATE}_full.txt
  done
done
