mkdir results
for N_PARTITIONS in 5 8 10
do
  for SAMPLING_RATE in 0.10 0.01 0.00
  do
    echo -e "\033[1mclean python processes\033[0m"
    sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
    echo -e "\033[1m${N_PARTITIONS} partitions, ${SAMPLING_RATE} sampling rate\033[0m"
    python main.py \
      --dataset ogbn-products \
      --dropout 0.3 \
      --lr 0.003 \
      --n-partitions ${N_PARTITIONS} \
      --n-epochs 500 \
      --model graphsage \
      --sampling-rate ${SAMPLING_RATE} \
      --n-layers 3 \
      --n-hidden 128 \
      --log-every 10 \
      --use-pp \
      |& tee results/ogbn-products_n${N_PARTITIONS}_p${SAMPLING_RATE}_full.txt
  done
done
