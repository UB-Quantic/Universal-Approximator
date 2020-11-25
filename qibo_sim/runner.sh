nohup taskset -c 0 python3 main.py --function relu --method cma --ansatz Fourier > fourier_cma_relu.out &
nohup taskset -c 1 python3 main.py --function poly --method cma --ansatz Fourier > fourier_cma_poly.out &
nohup taskset -c 2 python3 main.py --function step --method cma --ansatz Fourier > fourier_cma_step.out &
nohup taskset -c 4 python3 main.py --function tanh --method cma --ansatz Fourier > fourier_cma_tanh.out &

nohup taskset -c 5 python3 main.py --function relu --method cma --ansatz Weighted > weighted_cma_relu.out &
nohup taskset -c 6 python3 main.py --function poly --method cma --ansatz Weighted > weighted_cma_poly.out &
nohup taskset -c 7 python3 main.py --function step --method cma --ansatz Weighted > weighted_cma_step.out &
nohup taskset -c 8 python3 main.py --function tanh --method cma --ansatz Weighted > weighted_cma_tanh.out &

nohup taskset -c 9 python3 main_complex.py --modulus relu --phase relu --method cma --ansatz Weighted > weighted_cma_relu_relu.out &
nohup taskset -c 10 python3 main_complex.py --modulus relu --phase poly --method cma --ansatz Weighted > weighted_cma_relu_poly.out &
nohup taskset -c 11 python3 main_complex.py --modulus relu --phase step --method cma --ansatz Weighted > weighted_cma_relu_step.out &
nohup taskset -c 12 python3 main_complex.py --modulus relu --phase tanh --method cma --ansatz Weighted > weighted_cma_relu_tanh.out &


