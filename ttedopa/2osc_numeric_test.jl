#!/usr/bin/julia

using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV
using QuantumOptics

# Se lo script viene eseguito su Qtech, devo disabilitare l'output
# grafico altrimenti il programma si schianta.
if gethostname() == "qtech.fisica.unimi.it" ||
   gethostname() == "qtech2.fisica.unimi.it"
  ENV["GKSwstype"] = "100"
  @info "Esecuzione su server remoto. Output grafico disattivato."
else
  delete!(ENV, "GKSwstype")
  # Se la chiave "GKSwstype" non esiste non succede niente.
end

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")
include(lib_path * "/tedopa.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let
  @info "Caricamento dei parametri delle simulazioni."
  parameter_lists = load_parameters(ARGS)
  tot_sim_n = length(parameter_lists)

  # Se il primo argomento da riga di comando è una cartella (che dovrebbe
  # contenere i file dei parametri), mi sposto subito in tale posizione in modo
  # che i file di output, come grafici e tabelle, siano salvati insieme ai file
  # di parametri.
  prev_dir = pwd()
  if isdir(ARGS[1])
    cd(ARGS[1])
  end

  # Le seguenti liste conterranno i risultati della simulazione per ciascuna
  # lista di parametri fornita.
  timesteps_super = []
  occn_super = []
  spin_current_super = []
  bond_dimensions_super = []
  spin_chain_levels_super = []
  range_osc_left_super = []
  range_spins_super = []
  range_osc_right_super = []
  osc_chain_coefficients_left_super = []
  osc_chain_coefficients_right_super = []
  snapshot_super = []
  numeric_occn_super = []
  numeric_tsteps_super = []
  norm_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri tecnici
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    nquad = Int(parameters["PolyChaos_nquad"])

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    Ω = parameters["spectral_density_peak"]
    γ = parameters["spectral_density_half_width"]
    κ = parameters["spectral_density_overall_factor"]
    T = parameters["temperature"]
    ωc = parameters["frequency_cutoff"]
    oscdim = parameters["maximum_oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    n_osc_left = 2
    n_osc_right = 1

    # Definizione degli operatori nell'Hamiltoniana
    # =============================================
    # Calcolo dei coefficienti dalla densità spettrale
    @info "Simulazione $current_sim_n di $tot_sim_n: calcolo dei coefficienti TEDOPA."
    J(ω) = κ^2 * 0.5γ/π * (hypot(0.5γ, ω-Ω)^(-2) - hypot(0.5γ, ω+Ω)^(-2))
    Jtherm = ω -> thermalisedJ(J, ω, T)
    Jzero  = ω -> thermalisedJ(J, ω, 0)
    (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(Jtherm,
                                        (-ωc, 0, ωc),
                                        2;
                                        Nquad=nquad,
                                        discretization=lanczos)
    (Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(Jzero,
                                        (0, ωc),
                                        2;
                                        Nquad=nquad,
                                        discretization=lanczos)

    @info "Simulazione $current_sim_n di $tot_sim_n: calcolo della soluzione dell'equazione di Schrödinger."
    bosc = FockBasis(oscdim)
    bspin = SpinBasis(1//2)
    bcoll = tensor(bosc, bosc,
                   [bspin for _ ∈ 1:n_spin_sites]...,
                   bosc, bosc)

    # Costruisco i vari operatori per l'Hamiltoniano
    function num(i::Int)
      if i ∈ 1:2
        op = embed(bcoll, i, number(bosc))
      elseif i ∈ 2 .+ (1:n_spin_sites)
        op = embed(bcoll, i, 0.5*(sigmaz(bspin) + one(bspin)))
      elseif i ∈ 2 .+ n_spin_sites .+ (1:2)
        op = embed(bcoll, i, number(bosc))
      else
        throw(DomainError(i, "$i not in range"))
      end
      return op
    end

    hspinloc(i) = embed(bcoll, 2+i, sigmaz(bspin))
    hspinint(i) = tensor(one(bosc),
                         one(bosc),
                         [one(bspin) for i ∈ 1:i-1]...,
                         sigmap(bspin)⊗sigmam(bspin)+sigmam(bspin)⊗sigmap(bspin),
                         [one(bspin) for i ∈ 1:n_spin_sites-i-1]...,
                         one(bosc),
                         one(bosc))

    # Gli operatori Hamiltoniani
    # · oscillatori a sinistra
    Hoscsx = (Ωₗ[2] * num(1) +
              Ωₗ[1] * num(2) +
              κₗ[1] * tensor(create(bosc)⊗destroy(bosc)+destroy(bosc)⊗create(bosc),
                             [one(bspin) for _ ∈ 1:n_spin_sites]...,
                             one(bosc),
                             one(bosc)))

    # · interazione tra oscillatore a sinistra e spin
    Hintsx = ηₗ * tensor(one(bosc),
                         create(bosc)+destroy(bosc),
                         sigmax(bspin),
                         [one(bspin) for _ ∈ 1:n_spin_sites-1]...,
                         one(bosc),
                         one(bosc))

    # · catena di spin
    Hspin = 0.5ε*sum(hspinloc.(1:n_spin_sites)) -
            0.5*sum(hspinint.(1:n_spin_sites-1))

    # · interazione tra oscillatore a destra e spin
    Hintdx = ηᵣ * tensor(one(bosc),
                         one(bosc),
                         [one(bspin) for _ ∈ 1:n_spin_sites-1]...,
                         sigmax(bspin),
                         create(bosc)+destroy(bosc),
                         one(bosc))
    # · oscillatori a destra
    Hoscdx = (Ωᵣ[1] * num(2+n_spin_sites+1) +
              Ωᵣ[2] * num(2+n_spin_sites+2) +
              κᵣ[1] * tensor(one(bosc),
                             one(bosc),
                             [one(bspin) for _ ∈ 1:n_spin_sites]...,
                             create(bosc)⊗destroy(bosc)+destroy(bosc)⊗create(bosc)))

    H = Hoscsx + Hintsx + Hspin + Hintdx + Hoscdx

    # Stato iniziale (vuoto)
    ψ₀ = tensor(fockstate(bosc, 0),
                fockstate(bosc, 0),
                [spindown(bspin) for _ ∈ 1:n_spin_sites]...,
                fockstate(bosc, 0),
                fockstate(bosc, 0))

    function fout(t, ψ)
      occn = [real(QuantumOptics.expect(N, ψ))
               for N ∈ num.(1:(2+n_spin_sites+2))]
      return [occn; QuantumOptics.norm(ψ)]
    end
    tout, output = timeevolution.schroedinger(time_step_list, ψ₀, H; fout=fout)
    output = mapreduce(permutedims, vcat, output)
    numeric_occn = output[:, 1:end-1]
    statenorm = output[:, end]

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(numeric_occn_super, numeric_occn)
    push!(norm_super, statenorm)

    dict = Dict(:time => tout)
    sitelabels = ["L2"; "L1"; string.("S", 1:n_spin_sites); "R1"; "R2"]
    for (j, label) ∈ enumerate(sitelabels)
      push!(dict, Symbol(string("occn_", label)) => numeric_occn[:,j])
    end
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "_numeric.dat")
    CSV.write(filename, table)
  end

  @info "Creazione dei grafici con i risultati."
  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  plotsize = (600, 400)

  # Numeri di occupazione
  # ---------------------
  N = size(numeric_occn_super[begin], 2)-4
  plt = groupplot(timesteps_super,
                  numeric_occn_super,
                  parameter_lists;
                  labels=["L2" "L1" string.(1:N)... "R1" "R2"],
                  linestyles=[:dash :dash repeat([:solid], N)... :dash :dash],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (sol. numerica)",
                  plotsize=plotsize)
  savefig(plt, "occn_numeric.png")

  # Norma dello stato
  # -----------------
  plt = unifiedplot(timesteps_super,
                    norm_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"t",
                    ylabel=L"\Vert\psi(t)\Vert",
                    plottitle="Norma dello stato",
                    plotsize=plotsize)
  savefig(plt, "norm_numeric.png")

  # Numeri di occupazione (solo spin)
  # ---------------------------------
  plt = groupplot(timesteps_super,
                  [occn[:, 3:end-2] for occn ∈ numeric_occn_super],
                  parameter_lists;
                  rescale=false,
                  labels=reduce(hcat, string.("S", 1:N)),
                  linestyles=reduce(hcat, repeat([:solid], N)),
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (solo spin, sol. numerica)",
                  plotsize=plotsize)
  savefig(plt, "occn_spins_numeric.png")

  # Numeri di occupazione (oscillatori sx)
  # --------------------------------------
  data = [[occn[:, 1];;
           occn[:, 2];;
           occn[:, 1] .+ occn[: ,2]]
           for occn ∈ numeric_occn_super]
  plt = groupplot(timesteps_super,
                  data,
                  parameter_lists;
                  rescale=false,
                  labels=["L2" "L1" "L1+L2"],
                  linestyles=[:solid :solid :dash],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (osc sx, sol. numerica)",
                  plotsize=plotsize)
  savefig(plt, "occn_oscsx_numeric.png")
  
  # Numeri di occupazione (oscillatori dx)
  # --------------------------------------
  data = [[occn[:, end-1];;
           occn[:, end];;
           occn[:, end-1] .+ occn[:, end]]
           for occn ∈ numeric_occn_super]
  plt = groupplot(timesteps_super,
                  data,
                  parameter_lists;
                  rescale=false,
                  labels=["R1" "R2" "R1+R2"],
                  linestyles=[:solid :solid :dash],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (osc dx, sol. numerica)",
                  plotsize=plotsize)
  savefig(plt, "occn_oscdx_numeric.png")
 
  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
