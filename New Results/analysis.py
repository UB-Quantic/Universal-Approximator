import numpy as np
import matplotlib.pyplot as plt
import os 

def tanh(x):
    return ( (1 + np.tanh(x) ) / 2 )

def relu(x):
    return np.clip(x, 0, np.max(x))

_RELPATH = os.path.dirname(os.path.abspath(__file__))
_RESULTS_PATH = os.path.join(_RELPATH, "Param evolution")

if __name__ == "__main__":
    
    p2_relu_file = os.path.join(_RESULTS_PATH, "p2relu4.txt")
    p2_relu = np.loadtxt(p2_relu_file)
    plt.plot(p2_relu)
    plt.xlabel("Evaluations")
    plt.ylabel("Angle")
    plt.show()


    # tanh_2L_5k_31p = os.path.join(_RESULTS_PATH, "Exp_tanh_2L 5k 7n 31p chi Powell.txt")
    # tanh_2L_5k_31p_om = os.path.join(_RESULTS_PATH, "Exp_tanh_2L 5k 7n 31p om chi Powell.txt")
    # tanh_2L_10k_41p_cal25 = os.path.join(_RESULTS_PATH, "Exp_tanh_2L 10k 7n 41p chi cal25 Powell.txt")
    # tanh_2L_10k_41p = os.path.join(_RESULTS_PATH, "Exp_tanh_2L 10k 7n 41p chi Powell.txt")
    
    # tanh_4L_10k_41p = os.path.join(_RESULTS_PATH, "Exp_tanh_4L 10k 7n 41p chi Powell.txt")
    
    # tanh_2L_10k_41p_log = os.path.join(_RESULTS_PATH, "Exp_tanh_2L 10k 7n 41p log Powell.txt")
    # tanh_2L_50k = os.path.join(_RESULTS_PATH, "Exp_tanh_2L 50k 7n 41p chi Powell.txt")
 
    # relu_1L_10k_41p_prev = os.path.join(_RESULTS_PATH, "Exp_relu_1L 10k 7n 41p chi Powell p0prev.txt")
    # relu_2L_10k_41p_prev = os.path.join(_RESULTS_PATH, "Exp_relu_2L 10k 7n 41p chi Powell p0prev.txt")
    # relu_3L_10k_41p_prev = os.path.join(_RESULTS_PATH, "Exp_relu_3L 10k 7n 41p chi Powell p0prev.txt")
    # relu_4L_10k_41p_prev = os.path.join(_RESULTS_PATH, "Exp_relu_4L 10k 7n 41p chi Powell p0prev.txt")
 
 
    # relu_1L_th = os.path.join(_RESULTS_PATH, "The_relu_1L.txt")
    # relu_2L_th = os.path.join(_RESULTS_PATH, "The_relu_2L.txt")
    # relu_3L_th = os.path.join(_RESULTS_PATH, "The_relu_3L.txt")
    # relu_4L_th = os.path.join(_RESULTS_PATH, "The_relu_4L.txt")
    
    # data_tanh_2L_5k_31p = np.loadtxt(tanh_2L_5k_31p)
    # data_tanh_2L_5k_31p_om = np.loadtxt(tanh_2L_5k_31p_om)
    # data_tanh_2L_10k_41p_cal25 = np.loadtxt(tanh_2L_10k_41p_cal25)
    # data_tanh_2L_10k_41p = np.loadtxt(tanh_2L_10k_41p)

    # data_tanh_4L_10k_41p = np.loadtxt(tanh_4L_10k_41p)
    
    # data_tanh_2L_10k_41p_log = np.loadtxt(tanh_2L_10k_41p_log)
    # # data_tanh_2L_50k = np.loadtxt(tanh_2L_50k)
    # data_relu_1L_th = np.loadtxt(relu_1L_th)
    # data_relu_2L_th = np.loadtxt(relu_2L_th)
    # data_relu_3L_th = np.loadtxt(relu_3L_th)
    # data_relu_4L_th = np.loadtxt(relu_4L_th)


    # data_relu_1L_10k_41p_prev = np.loadtxt(relu_1L_10k_41p_prev)
    # data_relu_2L_10k_41p_prev = np.loadtxt(relu_2L_10k_41p_prev)
    # data_relu_3L_10k_41p_prev = np.loadtxt(relu_3L_10k_41p_prev)
    # data_relu_4L_10k_41p_prev = np.loadtxt(relu_4L_10k_41p_prev)

    # # plt.plot(data_tanh_2L_5k_31p[0],data_tanh_2L_5k_31p[1],  label="tanh_2L_5k_31p")
    # # plt.plot(data_tanh_2L_5k_31p_om[0],data_tanh_2L_5k_31p_om[1],  label="tanh_2L_5k_31p_om")
    # # plt.plot(data_tanh_2L_10k_41p_cal25[0],data_tanh_2L_10k_41p_cal25[1],  label="tanh_2L_10k_41p_cal25")
    # # plt.plot(data_tanh_2L_10k_41p[0],data_tanh_2L_10k_41p[1],  label="tanh_2L_10k_41p")
    
    # # plt.plot(data_tanh_4L_10k_41p[0],data_tanh_4L_10k_41p[1],  label="tanh_4L_10k_41p")
    
    # # plt.plot(data_tanh_2L_10k_41p_log[0],data_tanh_2L_10k_41p_log[1],  label="tanh_2L_10k_41p_log")
    # # plt.plot(data_tanh_2L_50k[0],data_tanh_2L_50k[1],  label="tanh_2L_50k")
    # # plt.plot(data_tanh_2L_th[0],data_tanh_2L_th[1], label="tanh_2L_th")
    # plt.plot(data_relu_1L_th[0], data_relu_1L_th[1],label="relu_1L_th")
    # plt.plot(data_relu_2L_th[0], data_relu_2L_th[1],label="relu_2L_th")
    # plt.plot(data_relu_3L_th[0], data_relu_3L_th[1],label="relu_3L_th")
    # plt.plot(data_relu_4L_th[0], data_relu_4L_th[1],label="relu_4L_th")
    
    # plt.plot(data_relu_4L_th[0], relu(data_relu_4L_th[0]), "--", label="relu")

    # # plt.plot(data_relu_1L_10k_41p_prev[0], data_relu_1L_10k_41p_prev[1], label="relu_1L_10k_41p_prev")
    # # plt.plot(data_relu_2L_10k_41p_prev[0], data_relu_2L_10k_41p_prev[1], label="relu_2L_10k_41p_prev")
    # # plt.plot(data_relu_3L_10k_41p_prev[0], data_relu_3L_10k_41p_prev[1], label="relu_3L_10k_41p_prev")
    # # plt.plot(data_relu_4L_10k_41p_prev[0], data_relu_4L_10k_41p_prev[1], label="relu_4L_10k_41p_prev")
    # plt.xlabel("x")
    # plt.ylabel("$P_e$")
    # plt.legend(loc=2)
    # plt.show()
