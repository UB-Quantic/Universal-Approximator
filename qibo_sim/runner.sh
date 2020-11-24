nohup taskset -c 4 python3 main.py --function relu --method cma --ansatz Fourier > fourier_cma_relu.out &
nohup taskset -c 5 python3 main.py --function poly --method cma --ansatz Fourier > fourier_cma_poly.out &
nohup taskset -c 9 python3 main.py --function step --method cma --ansatz Fourier > fourier_cma_step.out &
nohup taskset -c 12 python3 main.py --function tanh --method cma --ansatz Fourier > fourier_cma_tanh.out &


