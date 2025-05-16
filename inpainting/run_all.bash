#srun -t 240 --gres=gpu:1 bash ./inpaint_street.bash AMSTERDAM
#srun -t 240 --gres=gpu:1 bash ./inpaint_street.bash MADRID
#srun -t 240 --gres=gpu:1 bash ./inpaint_street.bash MADRID2
#srun -t 240 --gres=gpu:1 bash ./inpaint_street.bash PARIS
n=24
for i in $(seq 0 $((n-1))); do
    srun -t 240 --gres=gpu:1 bash ./inpaint_street.bash MIT $n $i &
    #echo "Iteration $i"
done
