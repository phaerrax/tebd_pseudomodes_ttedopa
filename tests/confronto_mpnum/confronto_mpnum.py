#coding = utf-8
#!/usr/bin/env python3

import sys
import getopt
import json
import csv
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import mpnum as mp
import mpnum.special as mpspecial
from scipy.linalg import expm
from tqdm import tqdm

# Comincio leggendo i parametri dal file JSON (che usa anche Julia)
def usage():
    print("Opzioni disponibili:\n\
  -p, --parameters=<file> specifica il file da cui leggere i parametri\n\
  -q, --quiet             non mostrare i grafici (opzionale)")

try:
    opts, args = getopt.getopt(sys.argv[1:], "qp:", ["--quiet","--parameters="])
except getopt.error as err:
    print(err)
    usage()
    sys.exit(2)

suppress_plots = False
if "-p" not in [o for (o, a) in opts] and\
        "--parameters" not in [o for (o, a) in opts]:
            print("L'opzione -p/--parameters è obbligatoria.")
            sys.exit(2)

for opt, arg in opts:
    if opt in ("-p", "--parameters"):
        input_filename = arg
        with open(input_filename, "r") as pfile:
            parameters = json.loads(pfile.read())
    if opt in ("-q", "--quiet"):
        suppress_plots = True

n_spin_sites = parameters["number_of_spin_sites"]
max_err = parameters["MP_compression_error"]
max_dim = parameters["MP_maximum_bond_dimension"]

epsilon = parameters["spin_excitation_energy"]
l = 1

time_step = parameters["simulation_time_step"]
end_time = parameters["simulation_end_time"]
time_slices = np.arange(0, end_time + time_step, time_step)
# Quel `+ time_step` mi serve perché la lista includa anche l'istante finale
# (cioè `end_time`). In questo modo però può anche capitare che la lista vada
# anche oltre quell'istante, perciò ora la filtro per eliminare eventuali
# istanti oltre il limite.
time_slices = list(filter(lambda t: t <= end_time, time_slices))
skip_steps = parameters["skip_steps"]

up = np.array([1, 0])
down = np.array([0, 1])
single_ex_states = [mp.MPArray.from_kron([down] * j + [up] + [down] * (n_spin_sites - 1 - j)) for j in range(n_spin_sites)]
initial_state = single_ex_states[0]

# Il passo successivo è costruire l'operatore di evoluzione temporale come MPO, da applicare in più turni seguendo l'approssimazione di Trotter al second'ordine.
# Il termine dell'Hamiltoniano che agisce sui siti primi vicini è uguale per tutte le coppie di siti adiacenti, quindi lo definisco solo una volta.

sigma_up   = np.array([[0, 1], [0, 0]])
sigma_down = np.array([[0, 0], [1, 0]])
sigma_z    = np.array([[1, 0], [0, -1]])

h1 = 0.5 * sigma_z
h2 = -0.5 * (np.kron(sigma_down, sigma_up) + (np.kron(sigma_up, sigma_down)))

local_cfs = list(repeat(epsilon, n_spin_sites))
local_cfs[0] *= 2
local_cfs[-1] *= 2
interaction_cfs = list(repeat(l, n_spin_sites-1))

h_list = [(0.5 * local_cfs[j] * np.kron(h1, np.eye(2)) +
           0.5 * local_cfs[j+1] * np.kron(np.eye(2), h1) +
           interaction_cfs[j] * h2) for j in range(n_spin_sites-1)]

u_odd = [expm(-0.5j * time_step * h) for h in h_list[0:n_spin_sites-1:2]]
u_even = [expm(-1j * time_step * h) for h in h_list[1:n_spin_sites-1:2]]

u_odd_mpo = [mp.MPArray.from_array_global(u.reshape(2, 2, 2, 2), ndims=2) for u in u_odd]
u_even_mpo = [mp.MPArray.from_array_global(u.reshape(2, 2, 2, 2), ndims=2) for u in u_even]

if n_spin_sites % 2 == 0:
    odd_links_mpo = mp.chain(u_odd_mpo)
    even_links_mpo = mp.chain([mp.eye(sites=1, ldim=2)] + u_even_mpo + [mp.eye(sites=1, ldim=2)])
else:
    odd_links_mpo = mp.chain(u_odd_mpo + [mp.eye(sites=1, ldim=2)])
    even_links_mpo = mp.chain([mp.eye(sites=1, ldim=2)] + u_even_mpo)

evolution_op = mp.dot(odd_links_mpo, mp.dot(even_links_mpo, odd_links_mpo))
evolution_op.compress(method='svd', relerr=max_err)

def occ_numbers(state):
    return np.abs([mp.inner(s, state).item()**2 for s in single_ex_states])

current_state = single_ex_states[0]
occ_n = [occ_numbers(current_state)]
ranks = [list(current_state.ranks)]

n_steps = len(time_slices)
# Con tutti gli elementi definiti finora calcolo dunque l'evoluzione dello stato iniziale.
with tqdm(range(1, n_steps)) as progress:
    for t in progress:
        progress.set_description("Evoluzione temporale")
        current_state = mp.dot(evolution_op, current_state)
        current_state.compress(method='svd', relerr=max_err, rank=max_dim)

        ranks += [list(current_state.ranks)]
        occ_n += [occ_numbers(current_state)]

if not suppress_plots:
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 24))
    axes = axes.flatten()

    ax = axes[0]
    for j, row in enumerate(np.transpose(occ_n)):
        label = "i" + str(j + 1)
        ax.plot(time_slices, row, label=label)
    ax.set_title("Numeri di occupazione dei siti")
    ax.set_xlabel("t")
    ax.legend()
    ax.set_ylabel("$|(s_j,\psi(t))|^2$")
    ax.grid(True)

    ax = axes[1]
    ax.plot(time_slices, ranks, label="mpnum")
    ax.set_title("Ranghi del MPS")
    ax.set_xlabel("t")
    ax.legend()
    ax.set_ylabel("$\chi_j$")
    ax.grid(True)

    plt.show()

output_filename = input_filename.replace(".json", "_python.csv")
with open(output_filename, "w", newline='\n') as csvfile:
    output_writer = csv.writer(csvfile, delimiter=',')
    output_writer.writerow(["time"] + \
                           ["occ_n"+str(j) for j in range(1,n_spin_sites+1)] + \
                           ["rank"+str(j) for j in range(1,n_spin_sites)])
    for j in range(len(time_slices)):
        output_writer.writerow([str(time_slices[j])] + \
                               [str(n) for n in occ_n[j]] + \
                               [str(r) for r in ranks[j]])
