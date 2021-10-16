#!/usr/bin/julia

using ITensors
using Plots
using LaTeXStrings
using ProgressMeter
using LinearAlgebra
using JSON

include("spin_chain_space.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let
  # Definizione degli operatori nell'equazione di Lindblad
  # ======================================================
  # Molti operatori sono già definiti in spin_chain_space.jl: rimangono quelli
  # particolari per questo sistema, cioè l'operatore quello per la coppia (1,2)
  function ITensors.op(::OpName"expL_sx", ::SiteType"vecS=1/2", s1::Index, s2::Index; t::Number, ε::Number, ξ::Number)
    L = ε * op("H1loc", s1) * op("id:id", s2) +
        0.5ε * op("id:id", s1) * op("H1loc", s2) +
        op("HspinInt", s1, s2) +
        ξ * op("damping", s1) * op("id:id", s2)
    return exp(t * L)
  end
  # e quello per la coppia (n-1,n)
  function ITensors.op(::OpName"expL_dx", ::SiteType"vecS=1/2", s1::Index, s2::Index; t::Number, ε::Number, ξ::Number)
    L = 0.5ε * op("H1loc", s1) * op("id:id", s2) +
        ε * op("id:id", s1) * op("H1loc", s2) +
        op("HspinInt", s1, s2) +
        ξ * op("id:id", s1) * op("damping", s2)
    return exp(t * L)
  end

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
  κ = parameters["damping_coefficient"]
  T = parameters["temperature"]

  # - discretizzazione dell'intervallo temporale
  total_time = parameters["simulation_end_time"]
  n_steps = Int(total_time * ε)
  time_step_list = collect(LinRange(0, total_time, n_steps))
  time_step = time_step_list[2] - time_step_list[1]
  # Al variare di ε devo cambiare anche n_steps se no non
  # tornano i risultati...

  # Costruzione della catena
  # ========================
  n_sites = parameters["number_of_spin_sites"] # per ora deve essere un numero pari
  # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
  sites = siteinds("vecS=1/2", n_sites)

  # Stati di singola eccitazione
  single_ex_states = MPS[productMPS(sites, n -> n == i ? "Up:Up" : "Dn:Dn") for i = 1:n_sites]

  # Costruzione dell'operatore di evoluzione
  # ========================================
  ξL = κ * (1 + 2 / (ℯ^(ε/T) - 1))
  ξR = κ

  links_odd = vcat(
    [op("expL_sx", sites[1], sites[2]; t=time_step, ε=ε, ξ=ξL)],
    [op("expHspin", sites[j], sites[j+1]; t=time_step, ε=ε) for j = 3:2:n_sites-3],
    [op("expL_dx", sites[n_sites-1], sites[n_sites]; t=time_step, ε=ε, ξ=ξR)]
  )
  links_even = [op("expHspin", sites[j], sites[j+1]; t=time_step, ε=ε) for j = 2:2:n_sites-2]

  #links_odd = vcat(
  #  [("expL_sx", (1, 2), (t=time_step, ε=ε, ξ=ξL,))],
  #  [("expL", (j, j+1), (t=time_step, ε=ε,)) for j = 3:2:n_sites-3],
  #  [("expL_dx", (n_sites-1, n_sites), (t=time_step, ε=ε, ξ=ξR,))]
  #)

  #links_even = [("expL", (j, j+1), (t=time_step, ε=ε,)) for j = 2:2:n_sites-2]

  #evol_op_odd_links = MPO(ops(links_odd, sites))
  #evol_op_even_links = MPO(ops(links_even, sites))

  # Osservabili da misurare
  # =======================
  # - la corrente di spin
  #   Il metodo per calcolarla è questo, finché non mi viene in mente
  #   qualcosa di più comodo: l'operatore della corrente,
  #   J_k = λ/2 (σ ˣ⊗ σ ʸ-σ ʸ⊗ σ ˣ),
  #   viene separato nei suoi due addendi, che sono applicati allo
  #   stato corrente in tempi diversi; in seguito sottraggo il secondo
  #   al primo, e moltiplico tutto per λ/2 (che è 1/2).
  function current_1(i::Int, obs_index::Int)
    if i == obs_index
      str = "vecσy"
    elseif i == obs_index+1
      str = "vecσx"
    else
      str = "vecid"
    end
    return str
  end
  function current_2(i::Int, obs_index::Int)
    if i == obs_index
      str = "vecσx"
    elseif i == obs_index+1
      str = "vecσy"
    else
      str = "vecid"
    end
    return str
  end
  #
  current_op_list_1 = MPS[productMPS(sites, n -> current_1(n, i)) for i = 1:n_sites-1]
  current_op_list_2 = MPS[productMPS(sites, n -> current_2(n, i)) for i = 1:n_sites-1]
  # Definisco la funzione che misura la corrente di uno stato MPS in modo
  # da poterla riusare dopo in maniera consistente
  # Restituisce una lista di n_sites-1 numeri che sono la corrente tra un
  # sito e il successivo.
  function measure_current(s::MPS)
    return [0.5 * real(inner(J1, s) - inner(J2, s)) for (J1, J2) in zip(current_op_list_1, current_op_list_2)]
  end

  # Simulazione
  # ===========
  # Stato iniziale: c'è un'eccitazione nel primo sito
  current_state = single_ex_states[1]

  # Misuro le osservabili sullo stato iniziale
  occ_n = [[inner(s, current_state) for s in single_ex_states]]
  maxdim_monitor = Int[maxlinkdim(current_state)]
  current = [measure_current(current_state)]

  # ...e si parte!
  progress = Progress(n_steps, 1, "Simulazione in corso ", 20)
  for step in time_step_list[2:end]
    current_state = apply(vcat(links_even, links_odd), current_state, cutoff=max_err, maxdim=32)
    #
    occ_n = vcat(occ_n, [[real(inner(s, current_state)) for s in single_ex_states]])
    current = vcat(current, [measure_current(current_state)])
    push!(maxdim_monitor, maxlinkdim(current_state))
    next!(progress)
  end

  # Grafici
  # =================================
  base_dir = "damped_chain/"
  # - grafico dei numeri di occupazione
  row = Vector{Float64}(undef, length(occ_n))
  for i = 1:length(occ_n)
    row[i] = occ_n[i][1]
  end
  occ_n_plot = plot(time_step_list, row, title="Numero di occupazione dei siti", label="1")
  for i = 2:n_sites
    for j = 1:length(occ_n)
      row[j] = occ_n[j][i]
    end
    plot!(occ_n_plot, time_step_list, row, label=string(i))
  end
  xlabel!(occ_n_plot, L"$\lambda\,t$")
  ylabel!(occ_n_plot, L"$\langle n_i\rangle$")
  savefig(occ_n_plot, base_dir * "occ_n.png")

  # - grafico dei ranghi dell'MPS
  maxdim_monitor_plot = plot(time_step_list, maxdim_monitor)
  xlabel!(maxdim_monitor_plot, L"$\lambda\,t$")
  savefig(maxdim_monitor_plot, base_dir * "maxdim_monitor.png")

  # - grafico della corrente di spin
  row = Vector{Float64}(undef, length(current))
  for i = 1:length(current)
    row[i] = current[i][1]
  end
  current_plot = plot(time_step_list, row, title="Corrente di spin", label="(1,2)")
  for i = 2:n_sites-1
    for j = 1:length(current)
      row[j] = current[j][i]
    end
    plot!(current_plot, time_step_list, row, label="("*string(i)*","*string(i+1)*")")
  end
  xlabel!(current_plot, L"$\lambda\,t$")
  ylabel!(current_plot, L"$j_{k,k+1}$")
  savefig(current_plot, base_dir * "spin_current.png")

  return
end
