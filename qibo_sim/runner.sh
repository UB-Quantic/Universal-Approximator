
nohup taskset -c 0 python3 main_2D.py --function adjiman --layers 6 --method cma > 24_12_0.out &
nohup taskset -c 1 python3 main_2D.py --function adjiman --layers 6 --method l-bfgs-b > 24_12_1.out &
nohup taskset -c 2 python3 main_2D.py --function adjiman --layers 6 --method bfgs > 24_12_2.out &

nohup taskset -c 3 python3 main_2D.py --function himmelblau --layers 4 --method cma > 24_12_3.out &
nohup taskset -c 4 python3 main_2D.py --function himmelblau --layers 4 --method l-bfgs-b > 24_12_4.out &
nohup taskset -c 5 python3 main_2D.py --function himmelblau --layers 4 --method bfgs > 24_12_5.out &

nohup taskset -c 6 python3 main_2D.py --function threehump --layers 6 --method cma > 24_12_6.out &
nohup taskset -c 7 python3 main_2D.py --function threehump --layers 6 --method l-bfgs-b > 24_12_7.out &
nohup taskset -c 8 python3 main_2D.py --function threehump --layers 6 --method bfgs > 24_12_8.out &

nohup taskset -c 9 python3 main_2D.py --function brent --layers 4 --method cma > 24_12_9.out &
nohup taskset -c 10 python3 main_2D.py --function brent --layers 4 --method l-bfgs-b > 24_12_10.out &
nohup taskset -c 11 python3 main_2D.py --function brent --layers 4 --method bfgs > 24_12_11.out &

nohup taskset -c 12 python3 main_2D.py --function brent --layers 5 --method cma > 24_12_12.out &
nohup taskset -c 13 python3 main_2D.py --function brent --layers 5 --method l-bfgs-b > 24_12_13.out
nohup taskset -c 14 python3 main_2D.py --function brent --layers 5 --method bfgs > 24_12_14.out &