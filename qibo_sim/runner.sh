
nohup taskset -c 0 python3 main_complex.py --modulus relu --phases 1 --method cma --ansatz Fourier > fourier_cma_relu_1.out &
nohup taskset -c 1 python3 main_complex.py --modulus relu --phases 2 --method cma --ansatz Fourier > fourier_cma_relu_2.out &
nohup taskset -c 2 python3 main_complex.py --modulus step --phases 1 --method cma --ansatz Fourier > fourier_cma_step_1.out &
nohup taskset -c 3 python3 main_complex.py --modulus step --phases 2 --method cma --ansatz Fourier > fourier_cma_step_2.out &

nohup taskset -c 4 python3 main_complex.py --modulus poly --phases 1 --method cma --ansatz Fourier > fourier_cma_poly_1.out &
nohup taskset -c 5 python3 main_complex.py --modulus poly --phases 2 --method cma --ansatz Fourier > fourier_cma_poly_2.out &
nohup taskset -c 6 python3 main_complex.py --modulus tanh --phases 1 --method cma --ansatz Fourier > fourier_cma_tanh_1.out &
nohup taskset -c 7 python3 main_complex.py --modulus tanh --phases 2 --method cma --ansatz Fourier > fourier_cma_tanh_2.out &

nohup taskset -c 8 python3 main_complex.py --modulus relu --phases 1 --method cma --ansatz Weighted > weighted_cma_relu_1.out &
nohup taskset -c 9 python3 main_complex.py --modulus relu --phases 2 --method cma --ansatz Weighted > weighted_cma_relu_2.out &
nohup taskset -c 10 python3 main_complex.py --modulus step --phases 1 --method cma --ansatz Weighted > weighted_cma_step_1.out &
nohup taskset -c 11 python3 main_complex.py --modulus step --phases 2 --method cma --ansatz Weighted > weighted_cma_step_2.out &
##
nohup taskset -c 12 python3 main_complex.py --modulus poly --phases 1 --method cma --ansatz Weighted > weighted_cma_poly_1.out &
nohup taskset -c 13 python3 main_complex.py --modulus poly --phases 2 --method cma --ansatz Weighted > weighted_cma_poly_2.out &
nohup taskset -c 14 python3 main_complex.py --modulus tanh --phases 1 --method cma --ansatz Weighted > weighted_cma_tanh_1.out &
nohup taskset -c 15 python3 main_complex.py --modulus tanh --phases 2 --method cma --ansatz Weighted > weighted_cma_tanh_2.out &

#nohup taskset -c 1 python3 main_complex.py --modulus poly --phase relu --method cma --ansatz Fourier > Fourier_cma_poly_relu.out &
#nohup taskset -c 2 python3 main_complex.py --modulus poly --phase poly --method cma --ansatz Fourier > Fourier_cma_poly_poly.out &
#nohup taskset -c 3 python3 main_complex.py --modulus poly --phase step --method cma --ansatz Fourier > Fourier_cma_poly_step.out &
#nohup taskset -c 4 python3 main_complex.py --modulus poly --phase tanh --method cma --ansatz Fourier > Fourier_cma_poly_tanh.out &

#nohup taskset -c 5 python3 main_complex.py --modulus step --phase relu --method cma --ansatz Fourier > Fourier_cma_step_relu.out &
#nohup taskset -c 6 python3 main_complex.py --modulus step --phase poly --method cma --ansatz Fourier > Fourier_cma_step_poly.out &
#nohup taskset -c 7 python3 main_complex.py --modulus step --phase step --method cma --ansatz Fourier > Fourier_cma_step_step.out &
#nohup taskset -c 8 python3 main_complex.py --modulus step --phase tanh --method cma --ansatz Fourier > Fourier_cma_step_tanh.out &
##
#nohup taskset -c 9 python3 main_complex.py --modulus tanh --phase relu --method cma --ansatz Fourier > Fourier_cma_tanh_relu.out &
#nohup taskset -c 10 python3 main_complex.py --modulus tanh --phase poly --method cma --ansatz Fourier > Fourier_cma_tanh_poly.out &
#nohup taskset -c 11 python3 main_complex.py --modulus tanh --phase step --method cma --ansatz Fourier > Fourier_cma_tanh_step.out &
#nohup taskset -c 12 python3 main_complex.py --modulus tanh --phase tanh --method cma --ansatz Fourier > Fourier_cma_tanh_tanh.out &

#nohup taskset -c 1 python3 main_complex.py --modulus relu --phase relu --method cma --ansatz Fourier > Fourier_cma_relu_relu.out &
#nohup taskset -c 2 python3 main_complex.py --modulus relu --phase poly --method cma --ansatz Fourier > Fourier_cma_relu_poly.out &
#nohup taskset -c 3 python3 main_complex.py --modulus relu --phase step --method cma --ansatz Fourier > Fourier_cma_relu_step.out &
#nohup taskset -c 4 python3 main_complex.py --modulus relu --phase tanh --method cma --ansatz Fourier > Fourier_cma_relu_tanh.out &

