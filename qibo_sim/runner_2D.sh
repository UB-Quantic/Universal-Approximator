nohup taskset -c 0 python3 main_2D.py --function tanh_2D --method cma > weighted_2d_cma_tanh_2D.out &
nohup taskset -c 1 python3 main_2D.py --function paraboloid --method cma > weighted_2d_cma_paraboloid.out &
nohup taskset -c 2 python3 main_2D.py --function hyperboloid --method cma > weighted_2d_cma_hyperboloid.out &
nohup taskset -c 3 python3 main_2D.py --function relu_2D --method cma > weighted_2d_cma_relu_2D.out &

nohup taskset -c 4 python3 main_2D.py --function tanh_2D --method l-bfgs-b > weighted_2d_lbfgsb_tanh_2D.out &
nohup taskset -c 5 python3 main_2D.py --function paraboloid --method l-bfgs-b > weighted_2d_lbfgsb_paraboloid.out &
nohup taskset -c 6 python3 main_2D.py --function hyperboloid --method l-bfgs-b > weighted_2d_lbfgsb_hyperboloid.out &
nohup taskset -c 7 python3 main_2D.py --function relu_2D --method l-bfgs-b > weighted_2d_lbfgsb_relu_2D.out &













