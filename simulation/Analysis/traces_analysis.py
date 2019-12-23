import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    with open('simulation/Analysis/Traces_Polarization_Simulation.txt',"r") as f:
        all_data=[x.split() for x in f.readlines()]
        t = all_data[3]
        pol_traces_unformatted = all_data[4:]

    times = [float(x) * 1e9 for x in t]
    p_traces = [] # probability traces
    for trace in pol_traces_unformatted:
        pol_trace = [float(x) for x in trace]
        p_trace = [ (1-x)/2 for x in pol_trace]
        p_traces.append (p_trace)
        
    time_out_of_ground = []
    for trace in p_traces:
        toog = 0 # time out of ground
        for time_stamp in trace:
            toog += time_stamp
        time_out_of_ground.append(toog)

    x_coordinates = np.linspace(0,len(time_out_of_ground))
    x_coordinates = [(x / 50) * 20 - 10 for x in x_coordinates]
    plt.figure(1)
    plt.plot(x_coordinates, time_out_of_ground)
    # plt.figure(2)
    # plt.plot(times, p_traces[:])
    plt.show()


            

