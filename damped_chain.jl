#!/usr/bin/julia

using ITensors
using Plots
using LaTeXStrings
using ProgressMeter

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let
  # Definizione degli operatori vettorizzati
  # ========================================
  ITensors.space(::SiteType"vecS=1/2") = 4

  ITensors.state(::StateName"Up:Up", ::SiteType"vecS=1/2") = [
    1 0 0 0
  ]
  ITensors.state(::StateName"Dn:Dn", ::SiteType"vecS=1/2") = [
    0 0 0 1
  ]

  ITensors.op(::OpName"id:id", ::SiteType"vecS=1/2") = [
    # (I₂:I₂)
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
  ]
  ITensors.op(::OpName"σz:id", ::SiteType"vecS=1/2") = [
    # (σ ᶻ:I₂)
    1 0  0  0
    0 1  0  0
    0 0 -1  0
    0 0  0 -1
  ]
  ITensors.op(::OpName"id:σz", ::SiteType"vecS=1/2") = [
    # (I₂:σ ᶻ)
    1  0 0  0
    0 -1 0  0
    0  0 1  0
    0  0 0 -1
  ]
  ITensors.op(::OpName"id:σ+", ::SiteType"vecS=1/2") = [
    # (I₂:σ ⁺)
    0 1 0 0
    0 0 0 0
    0 0 0 1
    0 0 0 0
  ]
  ITensors.op(::OpName"σ+:id", ::SiteType"vecS=1/2") = [
    # (σ ⁺:I₂)
    0 0 1 0
    0 0 0 1
    0 0 0 0
    0 0 0 0
  ]
  ITensors.op(::OpName"id:σ-", ::SiteType"vecS=1/2") = [
    # (I₂:σ ⁻)
    0 0 0 0
    1 0 0 0
    0 0 0 0
    0 0 1 0
  ]
  ITensors.op(::OpName"σ-:id", ::SiteType"vecS=1/2") = [
    # (σ ⁻:I₂)
    0 0 0 0
    0 0 0 0
    1 0 0 0
    0 1 0 0
  ]
  ITensors.op(::OpName"σx:σx", ::SiteType"vecS=1/2") = [
    # (σ ˣ:σ ˣ)
    0 0 0 1
    0 0 1 0
    0 1 0 0
    1 0 0 0
  ]

  # Impostazione dei parametri della simulazione
  # ============================================
  # Cartella che conterrà i file prodotti
  base_dir = "damped_chain/"

  max_err = 1e-10

  epsilon = 150
  # lambda = 1
  xi = 0

  total_time = 10
  n_steps = Int(2 * total_time * epsilon)
  time_step_list = collect(LinRange(0, total_time, n_steps))
  time_step = time_step_list[2] - time_step_list[1]
  # Al variare di epsilon devo cambiare anche n_steps se no non
  # tornano i risultati...

  # Costruzione della catena
  # ========================
  n_sites = 10 # per ora deve essere un numero pari
  # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
  sites = siteinds("vecS=1/2", n_sites)

  # Stati di singola eccitazione
  single_ex_states = MPS[productMPS(sites, n -> n == i ? "Up:Up" : "Dn:Dn") for i = 1:n_sites]

  # Costruzione dell'operatore di evoluzione
  # ========================================
  links_odd = ITensor[]

  s1 = sites[1]
  s2 = sites[2]
  L = im/4 * epsilon * (
    2 * (op("σz:id", s1) - op("id:σz", s1)) * op("id:id", s2) +
    op("id:id", s1) * (op("σz:id", s2) - op("id:σz", s2))
  ) - im/2 * (
    op("σ-:id", s1) * op("σ+:id", s2) +
    op("σ+:id", s1) * op("σ-:id", s2) -
    op("id:σ+", s1) * op("id:σ-", s2) -
    op("id:σ-", s1) * op("id:σ+", s2)
   ) + xi/2 * (op("σx:σx", s1) - op("id:id", s1)) * op("id:id", s2)
  push!(links_odd, exp(time_step * L))

  for j = 3:2:n_sites-3
    s1 = sites[j]
    s2 = sites[j+1]
    L = im/4 * epsilon * (
      (op("σz:id", s1) - op("id:σz", s1)) * op("id:id", s2) +
      op("id:id", s1) * (op("σz:id", s2) - op("id:σz", s2))
    ) - im/2 * (
      op("σ-:id", s1) * op("σ+:id", s2) +
      op("σ+:id", s1) * op("σ-:id", s2) -
      op("id:σ+", s1) * op("id:σ-", s2) -
      op("id:σ-", s1) * op("id:σ+", s2)
    )
    push!(links_odd, exp(time_step * L))
  end

  s1 = sites[end-1] # j = n_sites-1
  s2 = sites[end] # j = n_sites
  L = im/4 * epsilon * (
    (op("σz:id", s1) - op("id:σz", s1)) * op("id:id", s2) +
    2 * op("id:id", s1) * (op("σz:id", s2) - op("id:σz", s2))
  ) - im/2 * (
    op("σ-:id", s1) * op("σ+:id", s2) +
    op("σ+:id", s1) * op("σ-:id", s2) -
    op("id:σ+", s1) * op("id:σ-", s2) -
    op("id:σ-", s1) * op("id:σ+", s2)
  ) + xi/2 * op("id:id", s1) * (op("σx:σx", s2) - op("id:id", s2))
  push!(links_odd, exp(time_step * L))
  
  links_even = ITensor[]
  for j = 2:2:n_sites-2
    s1 = sites[j]
    s2 = sites[j+1]
    L = im/4 * epsilon * (
      (op("σz:id", s1) - op("id:σz", s1)) * op("id:id", s2) +
      op("id:id", s1) * (op("σz:id", s2) - op("id:σz", s2))
    ) - im/2 * (
      op("σ-:id", s1) * op("σ+:id", s2) +
      op("σ+:id", s1) * op("σ-:id", s2) -
      op("id:σ+", s1) * op("id:σ-", s2) -
      op("id:σ-", s1) * op("id:σ+", s2)
    )
    push!(links_even, exp(time_step * L))
  end

  time_evolution_oplist = vcat(links_odd, links_even)

  # Lo stato iniziale ha un'eccitazione nel primo sito
  current_state = single_ex_states[1]
  occ_n = [[inner(s, current_state) for s in single_ex_states]]
  maxdim_monitor = Int[maxlinkdim(current_state)]

  # ..via!
  progress = Progress(n_steps, 1, "Simulazione in corso ", 20)
  for step in time_step_list[2:end]
    current_state = apply(time_evolution_oplist, current_state; cutoff=max_err)
    occ_n = vcat(occ_n, [[real(inner(s, current_state)) for s in single_ex_states]])
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

  return
end
