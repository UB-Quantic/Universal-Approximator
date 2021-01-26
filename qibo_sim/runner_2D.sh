nohup taskset -c 0 python3 main_2D.py --function himmelblau --ansatz Fourier_2D  --method cma > Fourier_2d_cma_himmelblau.out &
nohup taskset -c 1 python3 main_2D.py --function brent --ansatz Fourier_2D  --method cma > Fourier_2d_cma_brent.out &
nohup taskset -c 2 python3 main_2D.py --function threehump --ansatz Fourier_2D  --method cma > Fourier_2d_cma_threehump.out &
nohup taskset -c 3 python3 main_2D.py --function adjiman --ansatz Fourier_2D  --method cma > Fourier_2d_cma_adjiman.out &

nohup taskset -c 4 python3 main_2D.py --function himmelblau --ansatz Fourier_2D  --method l-bfgs-b > Fourier_2d_l-bfgs-b_himmelblau.out &
nohup taskset -c 5 python3 main_2D.py --function brent --ansatz Fourier_2D  --method l-bfgs-b > Fourier_2d_l-bfgs-b_brent.out &
nohup taskset -c 6 python3 main_2D.py --function threehump --ansatz Fourier_2D  --method l-bfgs-b > Fourier_2d_l-bfgs-b_threehump.out &
nohup taskset -c 7 python3 main_2D.py --function adjiman --ansatz Fourier_2D  --method l-bfgs-b > Fourier_2d_l-bfgs-b_adjiman.out &














