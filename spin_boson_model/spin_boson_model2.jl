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
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/operators.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# Differisce da "chain_with_damped_oscillators" in quanto qui
# indago l'andamento di quantità diverse e lo confronto con una
# soluzione esatta (per un numero di spin sufficientemente basso).

let  
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
  occ_n_MPS_super = []
  normalisation_super = []
  bond_dimensions_super = []
  occ_n_numeric_super = []
  diff_occ_n_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ = parameters["oscillator_spin_interaction_coefficient"]
    γₗ = parameters["oscillator_damping_coefficient_left"]
    γᵣ = parameters["oscillator_damping_coefficient_right"]
    ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]

    # Leggo dal file temporaneo i dati prodotti dall'altro script.
    tmpfilename = replace(parameters["filename"], ".json" => ".dat.tmp")
    outfilename = replace(parameters["filename"], ".json" => ".dat")
    MPSdata = DataFrame(CSV.File(tmpfilename))

    time_step_list = MPSdata[:, :time]
    occ_n_MPS = hcat(MPSdata[:, :occ_n_left],
                     [MPSdata[:, Symbol("occ_n_spin$n")]
                      for n ∈ 1:n_spin_sites]...,
                     MPSdata[:, :occ_n_right])
    bond_dimensions = hcat([MPSdata[:, Symbol("bond_dim$n")]
                            for n ∈ 1:(n_spin_sites+1)]...)

    @info "Costruzione dell'equazione di Lindblad."
    # Calcolo la soluzione esatta, integrando l'equazione di Lindblad.
    bosc = FockBasis(osc_dim)
    bspin = SpinBasis(1//2)
    bcoll = tensor(bosc, repeat([bspin], n_spin_sites)..., bosc)

    function numop(i::Int)
      if i == 1 || i == n_spin_sites+2
        op = embed(bcoll, i, number(bosc))
      elseif i ∈ 1 .+ (1:n_spin_sites)
        op = embed(bcoll, i, QuantumOptics.projector(spinup(bspin)))
      else
        throw(DomainError)
      end
      return op
    end
    hspinloc(i) = embed(bcoll, 1+i, sigmaz(bspin))
    hspinint(i) = tensor(one(bosc),
                         repeat([one(bspin)], i-1)...,
                         tensor(sigmap(bspin), sigmam(bspin)) +
                           tensor(sigmam(bspin), sigmap(bspin)),
                         repeat([one(bspin)], n_spin_sites-i-1)...,
                         one(bosc))

    # Gli operatori Hamiltoniani
    Hoscsx = ω * numop(1)
    Hintsx = κ * tensor(create(bosc)+destroy(bosc),
                        sigmax(bspin),
                        repeat([one(bspin)], n_spin_sites-1)...,
                        one(bosc))
    Hspin = 0.5ε*sum(hspinloc.(1:n_spin_sites)) -
            0.5*sum(hspinint.(1:n_spin_sites-1))
    Hintdx = κ * tensor(one(bosc),
                        repeat([one(bspin)], n_spin_sites-1)...,
                        sigmax(bspin),
                        create(bosc)+destroy(bosc))
    Hoscdx = ω * numop(2+n_spin_sites)

    H = Hoscsx + Hintsx + Hspin + Hintdx + Hoscdx

    # Lo stato iniziale
    if T == 0
      matL = projector(fockstate(bosc, 0))
      n = 0
    else
      matL = thermalstate(ω*number(bosc), T)
      n = (ℯ^(ω / T) - 1)^(-1)
    end
    # Gli spin partono il primo su, gli altri (se ci sono) giù.
    ρ₀ = tensor(matL,
                QuantumOptics.projector(spindown(bspin)),
                repeat([QuantumOptics.projector(spindown(bspin))],
                       n_spin_sites-1)...,
                QuantumOptics.projector(fockstate(bosc, 0)))

    rates = [γₗ * (n+1), γₗ * n, γᵣ]
    jump = [embed(bcoll, 1, destroy(bosc)),
            embed(bcoll, 1, create(bosc)),
            embed(bcoll, 2+n_spin_sites, destroy(bosc))]

    function fout(t, rho)
      occ_n = [real(QuantumOptics.expect(N, rho))
               for N in numop.(1:(2+n_spin_sites))]
      return [occ_n..., QuantumOptics.tr(rho)]
    end

    @info "Integrazione dell'equazione di Lindblad in corso."
    τ = time_step_list[begin+1] - time_step_list[begin]
    tout, output = timeevolution.master(time_step_list,
                                        ρ₀, H, jump;
                                        rates=rates,
                                        fout=fout,
                                        dt=0.1τ)
    output = permutedims(hcat(output...))
    occ_n_numeric = real.(output[:,1:end-1])
    norm_numeric = real.(output[:,end])

    @info "Scrittura dei risultati su file."
    dict = Dict(:time => time_step_list)
    for (col, name) ∈ enumerate([:occ_n_left_MPS;
                                 [Symbol("occ_n_spin$(n)_MPS")
                                  for n ∈ 1:n_spin_sites];
                                 :occ_n_right_MPS])
      push!(dict, name => occ_n_MPS[:,col])
    end

    for (col, name) ∈ enumerate([:occ_n_left_numeric;
                                 [Symbol("occ_n_spin$(n)_numeric")
                                  for n ∈ 1:n_spin_sites];
                                 :occ_n_right_numeric])
      push!(dict, name => occ_n_numeric[:,col])
    end

    diff_occ_n = occ_n_numeric .- occ_n_MPS
    for (col, name) ∈ enumerate([:diff_occ_n_left;
                                  [Symbol("diff_occ_n_spin$n")
                                   for n ∈ 1:n_spin_sites];
                                  :diff_occ_n_right])
      push!(dict, name => diff_occ_n[:,col])
    end

    for (col, name) ∈ enumerate([Symbol("bond_dim$n")
                                  for n ∈ 1:n_spin_sites+1])
      push!(dict, name => bond_dimensions[:,col])
    end

    norm_MPS = MPSdata[:, :full_trace]
    push!(dict, :trace_MPS => norm_MPS)
    push!(dict, :trace_numeric => norm_numeric)

    CSV.write(outfilename, DataFrame(dict))

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list)
    push!(occ_n_MPS_super, occ_n_MPS)
    push!(occ_n_numeric_super, occ_n_numeric)
    push!(diff_occ_n_super, diff_occ_n)
    push!(bond_dimensions_super, bond_dimensions)
    push!(normalisation_super, norm_MPS)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  plotsize = (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

  @info "Creazione dei grafici."
  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  N = size(occ_n_MPS_super[begin], 2)
  plt = groupplot(timesteps_super,
                  occ_n_MPS_super,
                  parameter_lists;
                  labels=["L" string.(1:N-2)... "R"],
                  linestyles=[:dash repeat([:solid], N-2)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (sol. MPS)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_MPS.png")

  N = size(occ_n_numeric_super[begin], 2)
  plt = groupplot(timesteps_super,
                  occ_n_numeric_super,
                  parameter_lists;
                  labels=["L" string.(1:N-2)... "R"],
                  linestyles=[:dash repeat([:solid], N-2)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (sol. numerica)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_numeric.png")

  N = size(diff_occ_n_super[begin], 2)
  plt = groupplot(timesteps_super,
                  diff_occ_n_super,
                  parameter_lists;
                  labels=["L" string.(1:N-2)... "R"],
                  linestyles=[:dash repeat([:solid], N-2)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (diff. MPS vs numerica)",
                  plotsize=plotsize)

  savefig(plt, "diff_occ_n.png")

  #=
  # Grafico dell'occupazione del primo oscillatore (riunito)
  # -------------------------------------------------------
  # Estraggo da occ_n_super i valori dell'oscillatore sinistro.
  occ_n_osc_left_super = [occ_n[:,1] for occ_n in occ_n_MPS_super]
  plt = unifiedplot(timesteps_super,
                    occ_n_osc_left_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\langle n_L(t)\rangle",
                    plottitle="Occupazione dell'oscillatore sx",
                    plotsize=plotsize)

  savefig(plt, "occ_n_osc_left.png")

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  spinsonly = [mat[:, 2:end-1] for mat in occ_n_MPS_super]
  plt = groupplot(timesteps_super,
                  spinsonly,
                  parameter_lists;
                  labels=hcat(string.(1:N-2)...),
                  linestyles=hcat(repeat([:solid], N-2)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (solo spin)",
                  plotsize=plotsize)
  
  savefig(plt, "occ_n_spins_only.png")
  
  # Grafico dei numeri di occupazione (oscillatori + totale catena)
  # ---------------------------------------------------------------
  # sum(X, dims=2) prende la matrice X e restituisce un vettore colonna
  # le cui righe sono le somme dei valori sulle rispettive righe di X.
  sums = [[mat[:, 1] sum(mat[:, 2:end-1], dims=2) mat[:, end]]
          for mat in occ_n_super]
  plt = groupplot(timesteps_super,
                  sums,
                  parameter_lists;
                  labels=["L" "catena" "R"],
                  linestyles=[:solid :dot :solid],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (oscillatori + totale catena)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_sums.png")
  =#

  # Grafico dei ranghi del MPS
  # --------------------------
  N = size(bond_dimensions_super[begin])[2]
  plt = groupplot(timesteps_super,
                  bond_dimensions_super,
                  parameter_lists;
                  labels=hcat(["($j,$(j+1))" for j ∈ 1:N]...),
                  linestyles=hcat(repeat([:solid], N)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\chi_{k,k+1}(t)",
                  plottitle="Ranghi del MPS",
                  plotsize=plotsize)

  savefig(plt, "bond_dimensions.png")

  # Grafico della traccia della matrice densità
  # -------------------------------------------
  # Questo serve più che altro per controllare che rimanga sempre pari a 1.
  plt = unifiedlogplot(timesteps_super,
                       normalisation_super,
                       parameter_lists;
                       linestyle=:solid,
                       xlabel=L"\lambda\, t",
                       ylabel=L"\log\operatorname{tr}\,\rho(t)",
                       plottitle="Normalizzazione della matrice densità",
                       plotsize=plotsize)

  savefig(plt, "dm_normalisation.png")

  ## Grafico della corrente di spin
  ## ------------------------------
  #N = size(spin_current_super[begin])[2]
  #plt = groupplot(timesteps_super,
  #                spin_current_super,
  #                parameter_lists;
  #                labels=hcat(["($j,$(j+1))" for j ∈ 1:N]...),
  #                linestyles=hcat(repeat([:solid], N)...),
  #                commonxlabel=L"\lambda\, t",
  #                commonylabel=L"j_{k,k+1}(t)",
  #                plottitle="Corrente di spin",
  #                plotsize=plotsize)

  #savefig(plt, "spin_current.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
