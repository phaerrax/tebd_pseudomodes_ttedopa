#!/usr/bin/julia

using ITensors
using Plots

# Questo programma calcola l'evoluzione della catena di spin isolata,
# usando le tecniche dei MPS ed MPO.

let
  total_time = 50
  n_steps = 100
  time_step = total_time / n_steps

  max_err = 1e-8

  # Costruzione della catena
  # ========================
  n_sites = 10 # per ora deve essere un numero pari
  # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
  sites = siteinds("S=1/2", n_sites)
  # Lo stato iniziale ha un'eccitazione nel primo sito
  init_state = productMPS(sites, n -> n == 1 ? "Up" : "Dn")

  # Nuovi operatori di ITensors
  # ===========================
  # Identità su S=1/2
  ITensors.op(::OpName"id", ::SiteType"S=1/2") = [
    1 0
    0 1
  ]
    
  # Costruzione dell'operatore di evoluzione
  # ========================================
  epsilon = 150
  # lambda = 1
  # Ricorda:
  # - gli h_loc agli estremi della catena vanno trattati separatamente
  # - Sz è 1/2*sigma_z, la matrice sigma_z si chiama Z, e così per le
  #   altre matrici di Pauli
  links_odd = ITensor[]
  s1 = sites[1]
  s2 = sites[2]
  #h_loc = 1/2 * epsilon * op("Z", s1) * op("id", s2) +
  #        1/4 * epsilon * op("id", s1) * op("Z", s2) +
  h_loc = 1/2 * op("S+", s1) * op("S-", s2) +
          1/2 * op("S-", s1) * op("S+", s2)
  push!(links_odd, exp(-1.0im * time_step * h_loc))
  for j = 3:2:n_sites-3
    s1 = sites[j]
    s2 = sites[j+1]
    #h_loc = 1/4 * epsilon * op("Z", s1) * op("id", s2) +
    #        1/4 * epsilon * op("id", s1) * op("Z", s2) +
    h_loc = 1/2 * op("S+", s1) * op("S-", s2) +
            1/2 * op("S-", s1) * op("S+", s2)
    push!(links_odd, exp(-1.0im * time_step * h_loc))
  end
  s1 = sites[end-1] # j = n_sites-1
  s2 = sites[end] # j = n_sites
  #h_loc = 1/4 * epsilon * op("Z", s1) * op("id", s2) +
  #        1/2 * epsilon * op("id", s1) * op("Z", s2) +
  h_loc = 1/2 * op("S+", s1) * op("S-", s2) +
          1/2 * op("S-", s1) * op("S+", s2)
  push!(links_odd, exp(-1.0im * time_step * h_loc))
  
  links_even = ITensor[]
  for j = 2:2:n_sites-2
    s1 = sites[j]
    s2 = sites[j+1]
    #h_loc = 1/4 * epsilon * op("Z", s1) * op("id", s2) +
    #        1/4 * epsilon * op("id", s1) * op("Z", s2) +
    h_loc = 1/2 * op("S+", s1) * op("S-", s2) +
            1/2 * op("S-", s1) * op("S+", s2)
    push!(links_even, exp(-1.0im * time_step * h_loc))
  end

  time_evolution_oplist = vcat(links_odd, links_even)

  current_state = init_state
  sz = []
  for step = 1:n_steps
    current_state = apply(time_evolution_oplist, current_state; cutoff=max_err)
    append!(sz, abs(inner(init_state, current_state))^2)
  end

  plot(sz)
  png("sz.png")
  return
end
