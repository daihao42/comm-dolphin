#python train_magym.py --scenario traffic_junction --num-agent 10 --log-dir maddpg_ac128_traffic_finalreward_a10g14 --num-episodes 10000 --algorithm maddpg --max-episode-len 500 --lr 0.001
python train_magym.py --scenario traffic_junction --num-agent 10 --log-dir $1_ac128_traffic_finalreward_a10g14 --num-episodes 10000 --algorithm $1 --max-episode-len 500
