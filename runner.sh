#python train.py --scenario simple_spread --num-agent 20 --log-dir $1_ac128_fixlearnaction_finalreward_a20l20 --num-episodes 100000 --batch-size 64 --memory-capacity 1000 --algorithm $1
python train.py --scenario simple_spread --num-agent 7 --log-dir aamas_$1_ac128_fixlearnaction_finalreward_a7l7 --num-episodes 1000000 --batch-size 64 --memory-capacity 1000 --algorithm $1
