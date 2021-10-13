#!/usr/bin/julia

using ITensors
using Plots
using LaTeXStrings
using JSON

# Questo programma calcola l'evoluzione della catena di spin isolata,
# usando le tecniche dei MPS ed MPO.

let
  # Lettura dei parametri della simulazione
  # =======================================
  input_filename = ARGS[1]
  input = open(input_filename)
  s = read(input, String)
  parameters = JSON.parse(s)
  close(input)

  # - parametri per ITensors
  max_err = parameters["MP_compression_error"]
  max_dim = parameters["MP_maximum_bond_dimension"]

  # - parametri fisici
  ε = parameters["spin_excitation_energy"]
  # λ = 1

  # - discretizzazione dell'intervallo temporale
  total_time = parameters["simulation_end_time"]
  # Al variare di epsilon devo cambiare anche n_steps se no non
  # tornano i risultati... quindi lo definisco in seguito.

  # Costruzione della catena
  # ========================
  n_sites = parameters["number_of_spin_sites"] # per ora deve essere un numero pari
  # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
  sites = siteinds("S=1/2", n_sites)
  # Lo stato iniziale ha un'eccitazione nel primo sito
  init_state = productMPS(sites, n -> n == 1 ? "Up" : "Dn")

  # Stati di singola eccitazione
  single_ex_states = MPS[productMPS(sites, n -> n == i ? "Up" : "Dn") for i = 1:n_sites]

  # Nuovi operatori di ITensors
  # ===========================
  # Identità su S=1/2
  ITensors.op(::OpName"id", ::SiteType"S=1/2") = [
    1 0
    0 1
  ]
    
  # Costruzione dell'operatore di evoluzione
  # ========================================
  # In questo sistema semplice non sto usando la matrice identità e l'equazione
  # di Lindblad per calcolare la traiettoria del sistema, quindi non mi servono
  # le funzioni definite nei vari file ausiliari.
  n_steps = Int(total_time * ε)
  time_step_list = collect(LinRange(0, total_time, n_steps))
  time_step = time_step_list[2] - time_step_list[1]
  #
  # Ricorda:
  # - gli h_loc agli estremi della catena vanno trattati separatamente
  # - Sz è 1/2*sigma_z, la matrice sigma_z si chiama id, e così per le
  #   altre matrici di Pauli
  links_odd = ITensor[]
  s1 = sites[1]
  s2 = sites[2]
  h_loc = ε * op("Sz", s1) * op("id*id", s2) +
          1/2 * ε * op("id*id", s1) * op("Sz", s2) +
          -1/2 * op("S+", s1) * op("S-", s2) +
          -1/2 * op("S-", s1) * op("S+", s2)
  push!(links_odd, exp(-1.0im * time_step * h_loc))
  for j = 3:2:n_sites-3
    s1 = sites[j]
    s2 = sites[j+1]
    h_loc = 1/2 * ε * op("Sz", s1) * op("id*id", s2) +
            1/2 * ε * op("id*id", s1) * op("Sz", s2) +
            -1/2 * op("S+", s1) * op("S-", s2) +
            -1/2 * op("S-", s1) * op("S+", s2)
    push!(links_odd, exp(-1.0im * time_step * h_loc))
  end
  s1 = sites[end-1] # j = n_sites-1
  s2 = sites[end] # j = n_sites
  h_loc = 1/2 * ε * op("Sz", s1) * op("id*id", s2) +
          ε * op("id*id", s1) * op("Sz", s2) +
          -1/2 * op("S+", s1) * op("S-", s2) +
          -1/2 * op("S-", s1) * op("S+", s2)
  push!(links_odd, exp(-1.0im * time_step * h_loc))
  
  links_even = ITensor[]
  for j = 2:2:n_sites-2
    s1 = sites[j]
    s2 = sites[j+1]
    h_loc = 1/2 * ε * op("Sz", s1) * op("id*id", s2) +
            1/2 * ε * op("id*id", s1) * op("Sz", s2) +
            -1/2 * op("S+", s1) * op("S-", s2) +
            -1/2 * op("S-", s1) * op("S+", s2)
    push!(links_even, exp(-1.0im * time_step * h_loc))
  end

  time_evolution_oplist = vcat(links_odd, links_even)

  current_state = init_state
  occ_n = [[abs2(inner(s, current_state)) for s in single_ex_states]]
  maxdim_monitor = Int[]

  for step in time_step_list[2:end]
    current_state = apply(time_evolution_oplist, current_state; cutoff=max_err)
    occ_n = vcat(occ_n, [[abs2(inner(s, current_state)) for s in single_ex_states]])
    push!(maxdim_monitor, maxlinkdim(current_state))
  end

  # Grafici
  # =================================
  base_dir = "simple_chain/"
  # - grafico dei numeri di occupazione
  row = Vector{Float64}(undef, length(occ_n))
  for i = 1:length(occ_n)
    row[i] = occ_n[i][1]
  end
  occ_n_plot = plot(row, title="Numero di occupazione dei siti", label="1")
  for i = 2:n_sites
    for j = 1:length(occ_n)
      row[j] = occ_n[j][i]
    end
    plot!(occ_n_plot, row, label=string(i))
  end
  xlabel!(occ_n_plot, L"$\lambda\,t$")
  ylabel!(occ_n_plot, L"$\langle n_i\rangle$")
  savefig(occ_n_plot, base_dir * "occ_n.png")

  # - grafico dei ranghi dell'MPS
  maxdim_monitor_plot = plot(maxdim_monitor)
  xlabel!(maxdim_monitor_plot, L"$\lambda\,t$")
  savefig(maxdim_monitor_plot, base_dir * "maxdim_monitor.png")

  return
end
