nohup taskset -c 0 python3 main.py --function relu --method cma --ansatz Fourier > fourier_cma_relu.out &
nohup taskset -c 1 python3 main.py --function poly --method cma --ansatz Fourier > fourier_cma_poly.out &
nohup taskset -c 2 python3 main.py --function step --method cma --ansatz Fourier > fourier_cma_step.out &
nohup taskset -c 3 python3 main.py --function relu --method bfgs --ansatz Fourier > fourier_bfgs_relu.out &
nohup taskset -c 4 python3 main.py --function poly --method bfgs --ansatz Fourier > fourier_bfgs_poly.out &
nohup taskset -c 5 python3 main.py --function step --method bfgs --ansatz Fourier > fourier_bfgs_step.out &
nohup taskset -c 6 python3 main.py --function relu --method l-bfgs-b --ansatz Fourier > fourier_lbfgsb_relu.out &
nohup taskset -c 7 python3 main.py --function poly --method l-bfgs-b --ansatz Fourier > fourier_lbfgsb_poly.out &
nohup taskset -c 8 python3 main.py --function step --method l-bfgs-b --ansatz Fourier > fourier_lbfgsb_step.out &

