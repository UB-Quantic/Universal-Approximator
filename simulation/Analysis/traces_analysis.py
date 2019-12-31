import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    with open('simulation/Analysis/Traces_Polarization_Simulation_201P.txt',"r") as f:
        all_data=[x.split() for x in f.readlines()]
        t = all_data[3]
        pol_traces_unformatted = all_data[4:]

    times = [float(x) * 1e9 for x in t]
    p_traces = [] # probability traces
    for trace in pol_traces_unformatted:
        pol_trace = [float(x) for x in trace]
        p_trace = [ (1-x)/2 for x in pol_trace]
        p_traces.append (p_trace)
        
    time_around_equator = []
    for trace in p_traces:
        tae = 0 # time out of ground
        for time_stamp in trace:
            tae += 1 - 2 * np.abs(0.5 - time_stamp)
        time_around_equator.append(tae)

    x_coordinates = np.linspace(0,1,len(time_around_equator))
    # x_coordinates = [(x / 50) * 20 - 10 for x in x_coordinates]
    plt.figure(1)
    plt.plot(x_coordinates, time_around_equator)
    # plt.figure(2)
    # plt.plot(times, p_traces[:])
    plt.show()


    # with open('simulation/Analysis/Result_Experiment_201P.txt',"r") as f:
    #     all_data=[x.split() for x in f.readlines()]
    #     y_data = all_data[4:]

    # y = [float(x) for y_line in y_data for x in y_line]
    # y_max = np.amax(y)
    # y_min = np.amin(y)
    # A = abs(y_max - y_min) / 2
    # c = ( y_max + y_min ) / 2
    # P0 = ( y - (c - A) ) / (2 * A) 
    # x = np.linspace(0,1,len(y))
    # # plt.figure(1)
    # # plt.plot(x,P0)
    # # plt.show()
    

    # with open('simulation/Analysis/Polarization_Simulation_201P.txt',"r") as f:
    #     all_data=[x.split() for x in f.readlines()]
    #     pol_unformatted = all_data[4:]

    # P0_sim = [  (1+float(x))/2 for pol_line in pol_unformatted for x in pol_line ]
    
    # plt.figure(1)
    # plt.plot(x, abs(P0 - P0_sim))
    # plt.figure(2)
    # plt.plot(x, P0)
    # plt. plot(x, P0_sim)
    # plt.show()
