#!/usr/bin/julia

using ITensors
using Plots
using LaTeXStrings
using ProgressMeter
using LinearAlgebra
using JSON

include("spin_chain_space.jl")
include("harmonic_oscillator_space.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let  
  # Imposta la cartella di base per l'output
  # ========================================
  src_path = Base.source_path()
  base_path = src_path[findlast(isequal('/'), src_path)+1:end]
  if base_path[end-2:end] == ".jl"
    base_path = base_path[begin:end-3]
  end
  base_path *= '/'

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
  κ = parameters["oscillator_spin_interaction_coefficient"]
  γ = parameters["oscillator_damping_coefficient"]
  ω = parameters["oscillator_frequency"]
  T = parameters["temperature"]

  avg_occ_n(T::Number, ω::Number) = 1 / (ℯ^(ω / T) - 1)

  # - discretizzazione dell'intervallo temporale
  total_time = parameters["simulation_end_time"]
  n_steps = Int(total_time * ε)
  time_step_list = collect(LinRange(0, total_time, n_steps))
  time_step = time_step_list[2] - time_step_list[1]
  # Al variare di ε devo cambiare anche n_steps se no non
  # tornano i risultati...

  # Costruzione della catena
  # ========================
  n_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
  sites = vcat(
    [Index(osc_dim^2, "vecOsc,Site,n=L")],
    siteinds("vecS=1/2", n_sites),
    [Index(osc_dim^2, "vecOsc,Site,n=R")]
  )

  # Stati di singola eccitazione
  single_ex_state_tags = [vcat(
    ["vecid"],
    [i == n ? "Up:Up" : "Dn:Dn" for i = 1:n_sites],
    ["vecid"]
  ) for n = 1:n_sites]
  single_ex_states = [MPS(sites, tags) for tags in single_ex_state_tags]

  #= Definizione degli operatori nell'equazione di Lindblad
     ======================================================
     I siti del sistema sono numerati come segue:
     | 1 | 2 | ... | n_sites | n_sites+1 | n_sites+2 |
       ↑   │                        │          ↑
       │   └───────────┬────────────┘          │
       │               │                       │
       │        catena di spin                 │
     oscillatore sx                    oscillatore dx
     (Dato che non so come definire delle funzioni "op" che accettano Index di
     tipi diversi, vecS=1/2 e vecOsc in questo caso, non posso spostare anche
     questi due operatori ℓ_sx ed ℓ_dx nei file separati...)
  =#
  # - operatore per la coppia oscillatore-spin di sinistra
  sL = sites[1]
  s1 = sites[2]
  ℓ_sx = ω * op("H1loc", sL) * op("id:id", s1) +
         0.5ε * op("id:id", sL) * op("H1loc", s1) +
         im*κ * op("asum:id", sL) * op("σx:id", s1) +
         -im*κ* op("id:asum", sL)  * op("id:σx", s1) +
         γ * op("damping", sL; ω=ω, T=T) * op("id:id", s1)
  expℓ_sx = exp(0.5time_step * ℓ_sx)
  #
  # - e quello per la coppia oscillatore-spin di destra
  sn = sites[end-1]
  sR = sites[end]
  ℓ_dx = 0.5ε * op("H1loc", sn) * op("id:id", sR) +
         ω * op("id:id", sn) * op("H1loc", sR) +
         im*κ * op("σx:id", sn) * op("asumT:id", sR) +
         -im*κ * op("id:σx", sn) * op("id:asum", sR) +
         γ * op("id:id", sn) * op("damping", sR; ω=ω, T=0)
  expℓ_dx = exp(0.5time_step * ℓ_dx)

  # Costruzione dell'operatore di evoluzione
  # ========================================
  links_odd = vcat(
    [expℓ_sx],
    [op("expℓ", sites[j], sites[j+1]; t=0.5time_step, ε=ε) for j = 3:2:n_sites],
    [expℓ_dx]
  )
  links_even = [op("expℓ", sites[j], sites[j+1]; t=time_step, ε=ε) for j = 2:2:n_sites+1]

  # Osservabili da misurare
  # =======================
  # - i numeri di occupazione: per gli spin della catena si prende il prodotto
  #   interno con gli elementi di single_ex_states già definiti; per gli
  #   oscillatori, invece, uso
  osc_num_sx = MPS(sites, vcat(
    ["vecnum"],
    repeat(["vecid"], n_sites),
    ["vecid"]
  ))
  osc_num_dx = MPS(sites, vcat(
    ["vecid"],
    repeat(["vecid"], n_sites),
    ["vecnum"]
  ))
  occ_n_list = vcat([osc_num_sx], single_ex_states, [osc_num_dx])

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
  current_op_list_1 = [MPS(sites, vcat(
    ["Emp:Emp"],
    [current_1(i, k) for i = 1:n_sites],
    ["Emp:Emp"]
  )) for k = 1:n_sites-1]
  current_op_list_2 = [MPS(sites, vcat(
    ["Emp:Emp"],
    [current_2(i, k) for i = 1:n_sites],
    ["Emp:Emp"]
  )) for k = 1:n_sites-1]
  current_op_list = [0.5 * (J₊ - J₋) for (J₊,J₋) in zip(current_op_list_1, current_op_list_2)]
  # Ora per misurare la corrente basta prendere il prodotto interno con questi.

  # Simulazione
  # ===========
  # Stato iniziale: l'oscillatore sx è in equilibrio termico, il resto è vuoto
  function ITensors.state(::StateName"ThermEq", ::SiteType"vecOsc")
    mat = exp(-ω / T * num)
    mat /= tr(mat)
    return vcat(mat[:]) # (vcat impila le colonne)
  end
  if T == 0
    current_state = single_ex_states[1]
  else
    current_state = MPS(sites, vcat(
      ["ThermEq"],
      repeat(["Dn:Dn"], n_sites),
      ["Emp:Emp"]
    ))
  end

  # Misuro le osservabili sullo stato iniziale
  occ_n = [[inner(s, current_state) for s in occ_n_list]]
  maxdim_monitor = Int[maxlinkdim(current_state)]
  current = [[real(inner(j, current_state)) for j in current_op_list]]

  # ...e si parte!
  progress = Progress(n_steps, 1, "Simulazione in corso ", 20)
  for step in time_step_list[2:end]
    # Uso l'espansione di Trotter al 2° ordine
    current_state = apply(vcat(links_odd, links_even, links_odd), current_state, cutoff=max_err, maxdim=max_dim)
    #
    occ_n = vcat(occ_n, [[real(inner(s, current_state)) for s in occ_n_list]])
    current = vcat(current, [[real(inner(j, current_state)) for j in current_op_list]])
    push!(maxdim_monitor, maxlinkdim(current_state))
    next!(progress)
  end

  # Grafici
  # =======
  # - grafico dei numeri di occupazione
  row = Vector{Float64}(undef, length(occ_n))
  for j = 1:length(occ_n)
    row[j] = occ_n[j][1]
  end
  occ_n_plot = plot(time_step_list, row, title="Numero di occupazione dei siti", label="L", linestyle=:dash)
  for i = 1:n_sites
    for j = 1:length(occ_n)
      row[j] = occ_n[j][1+i]
    end
    plot!(occ_n_plot, time_step_list, row, label=string(i))
  end
  for j = 1:length(occ_n)
    row[j] = occ_n[j][end]
  end
  plot!(occ_n_plot, time_step_list, row, label="R", linestyle=:dash)
  #
  xlabel!(occ_n_plot, L"$\lambda\,t$")
  ylabel!(occ_n_plot, L"$\langle n_i\rangle$")
  savefig(occ_n_plot, base_path * "occ_n.png")

  # - grafico dei ranghi dell'MPS
  maxdim_monitor_plot = plot(time_step_list, maxdim_monitor)
  xlabel!(maxdim_monitor_plot, L"$\lambda\,t$")
  savefig(maxdim_monitor_plot, base_path * "maxdim_monitor.png")

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
  savefig(current_plot, base_path * "spin_current.png")

  return
end
