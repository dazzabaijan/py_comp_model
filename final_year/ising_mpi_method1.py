import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numba
import time
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size() #number of nodes including master
rank = comm.Get_rank()
name = MPI.Get_processor_name()
MASTER = 0

@numba.njit()
def sweep(lattice: np.ndarray, prob: np.ndarray, col_len_start: int,
          col_len_end: int, num_rows: int) -> np.ndarray:
    """
    Picks a spin at random from a LxL lattice with periodic boundary conditions
    Flip the spin according to importance sampling. 1 MC sweep = LxL flips.
    """
    for _ in range(num_rows*L):
        x, y = np.random.randint(col_len_start, col_len_end), np.random.randint(L)
        spin = lattice[x, y]

        up, down = (x-1) % L, (x+1) % L
        left, right = (y-1) % L, (y+1) % L

        nb_sum = lattice[up, y] + lattice[down, y] + \
                 lattice[x, left] + lattice[x, right]

        energy_diff = 2*spin*nb_sum

        if energy_diff < 0:
            spin *= -1
        elif np.random.rand() < prob[(int(energy_diff/4))]:
            spin *= -1

        lattice[x, y] = spin

    return lattice


@numba.njit()
def hamiltonian(lattice: np.ndarray) -> float:
    """
    Computes the Hamiltonian/energy of a lattice configuration by sum of
    nearest neighbours. At any point each spin has 4 neighbours, so dividing by
    4 takes into account of double counting.
    """
    energy = 0
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            spin = lattice[i, j]
            nb_sum = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + \
                     lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            energy += -nb_sum*spin

    return energy/4.

@numba.njit()
def subhamiltonian(lattice: np.ndarray, col_len_start: int,
                   col_len_end: int) -> float:

    energy = 0
    for i in range(col_len_start, col_len_end+1):
        for j in range(len(lattice)):
            spin = lattice[i, j]
            nb_sum = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + \
                     lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            energy += -nb_sum*spin

    return energy/4.


def metropolis(NT: int, L: int, EQ_STEPS: int, MC_STEPS: int, MAX_T: int,
                DS: int) -> float:
    start_time = time.time()
    np.random.seed(420)
    EQ_STEPS, MC_STEPS = np.uint64(int(EQ_STEPS)), np.uint64(int(MC_STEPS))
    MAX_T, NT, L = np.uint8(int(MAX_T)), np.uint16(int(NT)), np.uint16(int(L))
    T = np.linspace(1, MAX_T, NT)
    E, M = np.zeros(NT, dtype=np.float64), np.zeros(NT, dtype=np.float64)
    C, X = np.zeros(NT, dtype=np.float64), np.zeros(NT, dtype=np.float64)
    SCALE1 = DS/(MC_STEPS*L*L)
    SCALE2 = DS**2/(MC_STEPS*MC_STEPS*L*L)
    EQ_STEPS_NODE, MC_STEPS_NODE = int(EQ_STEPS/size), int(MC_STEPS/size)
    if (L-(size-1)) < size:
        raise ValueError("Lattice dimension too small for the available " +\
                         "processors to perform domain decomposition.")
    else:
        pass
    rows_per_node = (L-(size-1))//size
    rows_for_master = rows_per_node + (L-(size-1))%size
    sites_per_node = rows_per_node * L
    sites_for_master = rows_for_master * L
    for k in range(NT):
        E1 = E2 = M1 = M2 = np.float_(0.0)

        inv_T = 1.0 / T[k]  # beta = (1/k_B)*Temperature, set k_B = 1
        prob = np.exp(-4*inv_T*np.array([0,1,2], dtype=np.uint8))

        if (rank == MASTER):
            lattice = 2*np.random.randint(2, size=(L, L), dtype=np.int16)-1
            SCALE1 = DS/(MC_STEPS*L*L)
            SCALE2 = (DS)**2/(MC_STEPS*MC_STEPS*L*L)
            worker = size-1
            # plt.imshow(lattice, cmap='Greys', interpolation='nearest')
            # plt.show()
            start_index_master = (L - rows_for_master)
            end_index_master = (L-1)
            print(T[k])
            comm.bcast(lattice, root = MASTER)
            print(start_index_master)
            print(end_index_master)
            for i in range(EQ_STEPS_NODE):
                sweep(lattice, prob, start_index_master, end_index_master,
                      rows_for_master)
                if size > 1:
                    if i % sites_for_master == 0:
                        comm.Send([lattice[end_index_master-1:end_index_master], L, MPI.INT],
                                  dest = (rank+1)%size, tag=4)
                        comm.Recv([lattice[end_index_master-1:end_index_master], L, MPI.INT],
                                  source = (rank-1)%size, tag=4)
                        lattice = np.roll(lattice, 1, axis=0)
            for i in range(MC_STEPS_NODE):
                sweep(lattice, prob, start_index_master, end_index_master,
                      rows_for_master)
                if size > 1:
                    if i % sites_for_master == 0:
                        comm.Send([lattice[end_index_master-1:end_index_master], L, MPI.INT],
                                  dest = (rank+1)%size, tag=4)
                        comm.Recv([lattice[end_index_master-1:end_index_master], L, MPI.INT],
                                  source = (rank-1)%size, tag=4)
                        lattice = np.roll(lattice, 1, axis=0)

                if i % DS == 0:       # the remainder
                    energy = subhamiltonian(lattice, start_index_master, end_index_master)      # calculate the energy
                    mag = abs(np.sum(lattice[start_index_master:end_index_master+1]))     # calculate the magnetisation

                    E1 += size*energy
                    M1 += size*mag
                    E2 += size*energy*energy
                    M2 += size*size*mag*mag
                    print(energy)
            if size > 1:
                obsv = np.array([E1, M1, E2, M2])
                for node in range(worker):
                    obsv_node = np.zeros(4, dtype=np.float64)
                    comm.Recv(obsv_node, source = node+1, tag = 6)
                    obsv = obsv + obsv_node

                E[k] = SCALE1*obsv[0]
                M[k] = SCALE1*obsv[1]
                C[k] = (SCALE1*obsv[2] - SCALE2*obsv[0]*obsv[0])*inv_T*inv_T
                X[k] = (SCALE1*obsv[3] - SCALE2*obsv[1]*obsv[1])*inv_T
            else:
                E[k] = SCALE1*E1
                M[k] = SCALE1*M1
                C[k] = (SCALE1*E2 - SCALE2*E1*E1)*inv_T*inv_T
                X[k] = (SCALE1*M2 - SCALE2*M1*M1)*inv_T


            # plt.title(str(T[k]))
            # plt.imshow(lattice, cmap='Greys', interpolation='nearest')
            # plt.show()
        else: #Worker Code

            lattice = np.zeros((L,L))
            start_index = -1
            end_index = -2

            start_index = (rank-1)*(1+rows_per_node)
            end_index = start_index+rows_per_node
            lattice = comm.bcast(lattice, root = MASTER)      # receive broadcasted lattice from master
            for i in range(EQ_STEPS_NODE):
                sweep(lattice, prob, start_index, end_index, rows_per_node)

                if i % sites_for_master == 0:
                    comm.Send([lattice[end_index-1:end_index], L, MPI.INT],
                              dest=(rank+1)%size, tag=4)
                    comm.Recv([lattice[end_index-1:end_index], L, MPI.INT],
                              source = (rank-1)%size, tag=4)
                    lattice = np.roll(lattice, 1, axis=0)
#                    comm.Send([lattice[start_index:end_index], sites_per_node, MPI.INT], dest = MASTER, tag=5)
#                    lattice = comm.bcast(lattice, root = MASTER) # receive broadcast again from master
            for i in range(MC_STEPS_NODE):
                sweep(lattice, prob, start_index, end_index, rows_per_node)

                if i % sites_for_master == 0:
                    comm.Send([lattice[end_index-1:end_index], L, MPI.INT],
                              dest=(rank+1)%size, tag=4)
                    comm.Recv([lattice[end_index-1:end_index], L, MPI.INT],
                              source = (rank-1)%size, tag=4)
                    lattice = np.roll(lattice, 1, axis=0)

                if i % DS == 0:       # the remainder
                    energy = subhamiltonian(lattice, start_index, end_index)      # calculate the energy
                    mag = abs(np.sum(lattice[start_index:end_index+1]))     # calculate the magnetisation
#                    print(mag)
                    E1 += size*energy
                    M1 += size*mag
                    E2 += size*energy*energy
                    M2 += size*size*mag*mag

            obsv_node = np.array([E1, M1, E2, M2])
            comm.Send([obsv_node, 4, MPI.DOUBLE], dest = MASTER, tag = 6)


    total_time_elapsed = time.time() - start_time
    if rank==MASTER:
        print("(M)Total time elapsed: {0:.4f} seconds".format(total_time_elapsed))
        print('(M)Size of lattice\t\t\t: {}x{}'.format(L, L))
        print('(M)Temperature points\t\t\t: {} \t points'.format(NT))
        print('(M)Total MC sweeps for equilibration\t: {}  steps'.format(EQ_STEPS))
        print('(M)Total MC sweeps for calculation\t: {}  steps'.format(MC_STEPS))
        print('(M)MC sweeps for equilibration per node\t: {}  steps'.format(EQ_STEPS_NODE))
        print('(M)MC sweeps for calculation per node\t: {}  steps'.format(MC_STEPS_NODE))
        print('(M)Rows for master\t\t\t: {}'.format(rows_for_master))
        print('(M)Rows per node\t\t\t: {}'.format(rows_per_node))
        print('(M)Discarded samples\t\t\t: {} \tsamples'.format(DS))
        plt.figure(figsize=(18, 10))  # plot the calculated values
        y_data = [E, abs(M), C, X]
        color = ['IndianRed', 'RoyalBlue', 'IndianRed', 'RoyalBlue']
        ylabel = ["Energy", "Magnetisation", "Specific Heat", "Susceptibility"]
        df = pd.DataFrame({"Energy": E, "Magnetisation": M,
                           "Specific Heat": C, "Susceptibility": X})
        for i in range(0,4):
            plt.subplot(2, 2, i+1)
            plt.plot(T, y_data[i], marker='.', color=color[i])
            plt.xlabel("Temperature (T)", fontsize=20)
            plt.ylabel(ylabel[i], fontsize=20)
            plt.axis('tight')

        # plt.savefig(fname="L=" + str(L) + ",NT=" + str(NT) + ",DISC=" +
        #           str(DS) + "mpiworked.png")

        # df.to_csv("L=" + str(L)+",NT=" + str(NT) + ",DISC=" +
        #           str(DS) + ".csv", header=True, index=True)
    # else:
        # print("(W) rank {}".format(rank))
        # print("(W) size {}".format(size))
    return T, E, M, C, X

# NT = 50
# L = 3
# EQ_STEPS = (2**(L*L))
# MC_STEPS = 100*(2**(L*L))
# MAX_T = 20
# DS = 100

NT = 100
L = 128
EQ_STEPS = 5000
MC_STEPS = 100000
MAX_T = 4
DS = 10


start_time = time.time()
T, E, M, C, X = metropolis(NT, L, EQ_STEPS, MC_STEPS, MAX_T, DS)
time_elapsed = time.time() - start_time

if rank==MASTER:
    print("Total time elapsed: {} seconds".format(time_elapsed))

plt.show()
'''
np.savetxt("E"+str(L)+".csv", E, delimiter=",")
np.savetxt("M"+str(L)+".csv", M, delimiter=",")
np.savetxt("C"+str(L)+".csv", C, delimiter=",")
np.savetxt("X"+str(L)+".csv", X, delimiter=",")'''
