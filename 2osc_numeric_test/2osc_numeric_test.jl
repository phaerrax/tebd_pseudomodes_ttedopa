#!/usr/bin/julia

#using ITensors
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
  occ_n_super = []
  spin_current_super = []
  bond_dimensions_super = []
  spin_chain_levels_super = []
  range_osc_left_super = []
  range_spins_super = []
  range_osc_right_super = []
  osc_chain_coefficients_left_super = []
  osc_chain_coefficients_right_super = []
  snapshot_super = []
  numeric_occ_n_super = []
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
    # La densità spettrale è data da
    # J(ω) = κγ/π ⋅ 1/(γ² + (ω-Ω)²)
    T = parameters["temperature"]
    ωc = parameters["frequency_cutoff"]
    osc_dim = parameters["oscillator_space_dimension"]

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
    J(ω) = κ * γ/π * (1 / (γ^2 + (ω-Ω)^2) - 1 / (γ^2 + (ω+Ω)^2))
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
    #(Ωₗ, κₗ, ηₗ) = ([2.455, -2.186],
    #                [9.717],
    #                sqrt(quadgk(Jtherm, -ωc, 0, ωc)[1]))
    #(Ωᵣ, κᵣ, ηᵣ) = ([10.051],
    #                [1.375],
    #                sqrt(quadgk(Jzero, 0, ωc)[1]))

    @info "Simulazione $current_sim_n di $tot_sim_n: calcolo della soluzione dell'equazione di Schrödinger."
    bosc = FockBasis(osc_dim)
    bspin = SpinBasis(1//2)
    bcoll = tensor([bosc for i ∈ 1:2]...,
                   [bspin for i ∈ 1:n_spin_sites]...,
                   bosc)
    # Costruisco i vari operatori per l'Hamiltoniano
    function num(i::Int)
      if i ∈ 1:2
        op = embed(bcoll, i, number(bosc))
      elseif i ∈ 2 .+ (1:n_spin_sites)
        op = embed(bcoll, i, 0.5*(sigmaz(bspin) + one(bspin)))
      elseif i == 2 + n_spin_sites + 1
        op = embed(bcoll, i, number(bosc))
      else
        throw(DomainError)
      end
      return op
    end
    hspinloc(i) = embed(bcoll, 2+i, sigmaz(bspin))
    hspinint(i) = tensor(one(bosc),
                         one(bosc),
                         [one(bspin) for i ∈ 1:i-1]...,
                         sigmap(bspin)⊗sigmam(bspin)+sigmam(bspin)⊗sigmap(bspin),
                         [one(bspin) for i ∈ 1:n_spin_sites-i-1]...,
                         one(bosc))

    # Gli operatori Hamiltoniani
    # · oscillatori a sinistra
    Hoscsx = Ωₗ[1] * num(2) + Ωₗ[2] * num(1) +
             κₗ[1] * tensor(create(bosc)⊗destroy(bosc)+destroy(bosc)⊗create(bosc),
                            [one(bspin) for i ∈ 1:n_spin_sites]...,
                            one(bosc))

    # · interazione tra oscillatore a sinistra e spin
    Hintsx = ηₗ * tensor(one(bosc),
                         create(bosc)+destroy(bosc),
                         sigmax(bspin),
                         [one(bspin) for i ∈ 1:n_spin_sites-1]...,
                         one(bosc))

    # · catena di spin
    Hspin = 0.5ε*sum(hspinloc.(1:n_spin_sites)) -
            0.5*sum(hspinint.(1:n_spin_sites-1))
    # · interazione tra oscillatore a destra e spin
    Hintdx = ηᵣ * tensor(one(bosc),
                         one(bosc),
                         [one(bspin) for i ∈ 1:n_spin_sites-1]...,
                         sigmax(bspin),
                         create(bosc)+destroy(bosc))
    # · oscillatore a destra
    Hoscdx = Ωᵣ[1] * num(2+n_spin_sites+1)

    H = Hoscsx + Hintsx + Hspin + Hintdx + Hoscdx

    # Stato iniziale (vuoto)
    ψ₀ = tensor(fockstate(bosc, 0),
                fockstate(bosc, 0),
                [spindown(bspin) for i ∈ 1:n_spin_sites]...,
                fockstate(bosc, 0))

    function fout(t, psi)
      occ_n = [real(QuantumOptics.expect(N, psi))
               for N in num.(1:(2+n_spin_sites+1))]
      return [occ_n..., QuantumOptics.norm(psi)]
    end
    tout, output = timeevolution.schroedinger(time_step_list, ψ₀, H; fout=fout)
    output = permutedims(hcat(output...))
    numeric_occ_n = output[:,1:end-1]
    state_norm = output[:,end]
#=
    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto; mi riservo di decidere
    # volta per volta come inizializzare l'oscillatore più a destra nella
    # catena a sinistra, per motivi di diagnostica.
    osc_sx_init_state = chain(MPS(sites[1:n_osc_left-1], "0"),
                              parse_init_state_osc(
                                    sites[n_osc_left],
                                    parameters["left_oscillator_initial_state"]
                                   ))
    spin_init_state = parse_init_state(sites[range_spins],
                                       parameters["chain_initial_state"])
    osc_dx_init_state = MPS(sites[range_osc_right], "0")
    current_state = chain(osc_sx_init_state,
                          spin_init_state,
                          osc_dx_init_state)

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = [expect(current_state, "N")]
    bond_dimensions = [linkdims(current_state)]
    spin_current = [[real(inner(current_state, j * current_state))
                     for j in spin_current_ops]]
    spin_chain_levels = [levels(num_eigenspace_projs,
                                current_state)]

    # Evoluzione temporale
    # --------------------
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state;
                            cutoff=max_err,
                            maxdim=max_dim)
      if skip_count % skip_steps == 0
        push!(occ_n,
              expect(current_state, "N"))
        push!(spin_current,
              [real(inner(current_state, j * current_state))
               for j in spin_current_ops])
        push!(spin_chain_levels,
              levels(num_eigenspace_projs, current_state))
        push!(bond_dimensions,
              linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    snapshot = occ_n[end]

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => time_step_list[1:skip_steps:end])
    tmp_list = hcat(occ_n...)
    for (j, name) in enumerate([[Symbol("occ_n_left$n") for n∈n_osc_left:-1:1];
                                [Symbol("occ_n_spin$n") for n∈1:n_spin_sites];
                                [Symbol("occ_n_right$n") for n∈1:n_osc_right]])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(spin_current...)
    for (j, name) in enumerate([Symbol("spin_current$n")
                                for n = 1:n_spin_sites-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(spin_chain_levels...)
    for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_spin_sites])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(bond_dimensions...)
    len = n_osc_left + n_spin_sites + n_osc_right
    for (j, name) in enumerate([Symbol("bond_dim$n") for n ∈ 1:len-1])
      push!(dict, name => tmp_list[j,:])
    end
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)
    =#

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(numeric_occ_n_super, numeric_occ_n)
    push!(norm_super, state_norm)
    #push!(occ_n_super,
    #      permutedims(hcat(occ_n...)))
    #push!(spin_current_super,
    #      permutedims(hcat(spin_current...)))
    #push!(spin_chain_levels_super,
    #      permutedims(hcat(spin_chain_levels...)))
    #push!(bond_dimensions_super,
    #      permutedims(hcat(bond_dimensions...)))
    #push!(range_osc_left_super,
    #      range_osc_left)
    #push!(range_spins_super,
    #      range_spins)
    #push!(range_osc_right_super,
    #      range_osc_right)
    #push!(osc_chain_coefficients_left_super,
    #      osc_chain_coefficients_left)
    #push!(osc_chain_coefficients_right_super,
    #      osc_chain_coefficients_right)
    #push!(snapshot_super, snapshot)
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

  N = size(numeric_occ_n_super[begin], 2)
  plt = groupplot(timesteps_super,
                  numeric_occ_n_super,
                  parameter_lists;
                  labels=["L1" "L0" string.(1:N-3)... "R"],
                  linestyles=[:dash :dash repeat([:solid], N-2)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (sol. numerica)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_numeric.png")

  plt = unifiedplot(timesteps_super,
                    norm_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\Vert\psi(t)\Vert",
                    plottitle="Norma dello stato",
                    plotsize=plotsize)

  savefig(plt, "state_norm.png")

  #=
  # Grafico dei numeri di occupazione (oscillatori sx)
  # --------------------------------------------------
  plt = groupplot(timesteps_super,
                  [mat[:, range]
                   for (mat, range) ∈ zip(occ_n_super, range_osc_left_super)],
                  parameter_lists;
                  labels=[hcat(string.(reverse(range))...)
                          for range in range_osc_left_super],
                  linestyles=[hcat([:solid for i ∈ range]...)
                          for range in range_osc_left_super],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione "*
                             "(oscillatori a sx)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_osc_left.png")

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  plt = groupplot(timesteps_super,
                  [mat[:, range]
                   for (mat, range) ∈ zip(occ_n_super, range_spins_super)],
                  parameter_lists;
                  labels=[hcat(string.(range)...)
                          for range in range_spins_super],
                  linestyles=[hcat([:solid for i ∈ range]...)
                              for range in range_spins_super],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (spin)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_spins.png")
  
  # Grafico dei numeri di occupazione (oscillatori dx)
  # --------------------------------------------------
  plt = groupplot(timesteps_super,
                  [mat[:, range]
                   for (mat, range) ∈ zip(occ_n_super, range_osc_right_super)],
                  parameter_lists;
                  labels=[hcat(string.(range)...)
                          for range in range_osc_right_super],
                  linestyles=[hcat([:solid for i ∈ range]...)
                              for range ∈ range_osc_right_super],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione "*
                             "(oscillatori dx)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_osc_right.png")

  # Grafico dei numeri di occupazione (tot oscillatori + tot catena)
  # ----------------------------------------------------------------
  sums = [[sum(mat[:, rangeL], dims=2) sum(mat[:, rangeS], dims=2) sum(mat[:, rangeR], dims=2) sum(mat, dims=2)]
          for (mat, rangeL, rangeS, rangeR) in zip(occ_n_super,
                                                   range_osc_left_super,
                                                   range_spins_super,
                                                   range_osc_right_super)]
  plt = groupplot(timesteps_super,
                  sums,
                  parameter_lists;
                  labels=["osc. sx" "catena" "osc. dx" "tutti"],
                  linestyles=hcat(repeat([:solid], 4)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (sommati)",
                  plotsize=plotsize)
                  
  savefig(plt, "occ_n_sums.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  plt = groupplot(timesteps_super,
                  bond_dimensions_super,
                  parameter_lists;
                  labels=[hcat(["($j,$(j+1))" for j ∈ 1:size(v, 2)]...)
                          for v in bond_dimensions_super],
                  linestyles=[hcat(repeat([:solid], size(v, 2))...)
                              for v in bond_dimensions_super],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\chi_{k,k+1}(t)",
                  plottitle="Ranghi del MPS",
                  plotsize=plotsize)

  savefig(plt, "bond_dimensions.png")

  # Grafico della corrente di spin
  # ------------------------------
  plt = groupplot(timesteps_super,
                  spin_current_super,
                  parameter_lists;
                  labels=[hcat(["($j,$(j+1))" for j ∈ 1:size(c, 2)]...)
                          for c in spin_current_super],
                  linestyles=[hcat(repeat([:solid], size(c, 2))...)
                              for c in spin_current_super],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"j_{k,k+1}(t)",
                  plottitle="Corrente di spin",
                  plotsize=plotsize)

  savefig(plt, "spin_current.png")
 
  # Grafico dell'occupazione degli autospazi di N della catena di spin
  # ------------------------------------------------------------------
  # L'ultimo valore di ciascuna riga rappresenta la somma di tutti i
  # restanti valori.
  plt = groupplot(timesteps_super,
                  spin_chain_levels_super,
                  parameter_lists;
                  labels=[[string.( 0:(size(c,2)-2) )... "total"]
                          for c ∈ spin_chain_levels_super],
                  linestyles=[[repeat([:solid], size(c,2)-1)... :dash]
                              for c ∈ spin_chain_levels_super],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"n(t)",
                  plottitle="Occupazione degli autospazi "*
                            "della catena di spin",
                  plotsize=plotsize)

  savefig(plt, "chain_levels.png")

  # Grafico dei coefficienti della chain map
  # ----------------------------------------
  osc_sites = [reverse(1:length(chain[:,1]))
               for chain ∈ osc_chain_coefficients_left_super]
  plt = groupplot(osc_sites,
                  osc_chain_coefficients_left_super,
                  parameter_lists;
                  labels=[L"\Omega_i" L"\kappa_i"],
                  linestyles=[:solid :solid],
                  commonxlabel=L"i",
                  commonylabel="Coefficiente",
                  plottitle="Coefficienti della catena di "*
                             "oscillatori (sx)",
                  plotsize=plotsize)

  savefig(plt, "osc_left_coefficients.png")

  osc_sites = [1:length(chain[:,1])
               for chain ∈ osc_chain_coefficients_right_super]
  plt = groupplot(osc_sites,
                  osc_chain_coefficients_right_super,
                  parameter_lists;
                  labels=[L"\Omega_i" L"\kappa_i"],
                  linestyles=[:solid :solid],
                  commonxlabel=L"i",
                  commonylabel="Coefficiente",
                  plottitle="Coefficienti della catena di "*
                            "oscillatori (dx)",
                  plotsize=plotsize)

  savefig(plt, "osc_right_coefficients.png")

  # Istantanea dei numeri di occupazione alla fine
  # ----------------------------------------------
  plt = groupplot([[reverse(rangeL); rangeS; rangeR]
                     for (rangeL, rangeS, rangeR) ∈ zip(range_osc_left_super,
                                                        range_spins_super,
                                                        range_osc_right_super)],
                    snapshot_super,
                    parameter_lists;
                    labels=[L"\langle n_i\rangle"],
                    linestyles=[:solid],
                    commonxlabel=L"i",
                    commonylabel="Numero di occupazione",
                    plottitle="Numeri di occupazione alla fine",
                    plotsize=plotsize)

  savefig(plt, "snapshot.png")
  =#

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
