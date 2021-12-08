#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/tedopa.jl")
include(lib_path * "/operators_pytedopa.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

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
  occ_n_super = []
  coherence_super = []
  bond_dimensions_super = []
  osc_chain_coefficients_left_super = []
  snapshot_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri tecnici
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    nquad = Int(parameters["PolyChaos_nquad"])

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    λ = parameters["spin_interaction_energy"]
    Ω = parameters["ohmic_decay_factor"]
    a = parameters["ohmic_exponent"]
    b = parameters["ohmic_mult_factor"]
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
    n_osc_left = parameters["number_of_oscillators_left"]
    sites = [siteinds("Osc", n_osc_left; dim=osc_dim);
             siteinds("S=1/2", n_spin_sites)]

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#
    # Calcolo dei coefficienti dalla densità spettrale
    @info "Calcolo dei parametri TEDOPA in corso."
    support = (0, ωc)
    J(ω) = b * ℯ^(-ω/Ω) * (ω/Ω)^a

    #(Ωₗ, κₗ, ηₗ) = chainmapcoefficients(J, support, ωc, n_osc_left; Nquad=nquad)
    α = parse.(Float64, readlines("alphas.dat"))
    β = parse.(Float64, readlines("betas.dat"))
    #(Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(Jzero,
    #                                     (0, ωc),
    #                                     n_osc_right;
    #                                     Nquad=nquad)
    Ωₗ = ωc .* α[1:n_osc_left]
    κₗ = ωc .* sqrt.(β[2:n_osc_left])
    ηₗ = sqrt(β[1])
    #open("omegas_julia.dat", "w") do f
    #  for i in Ωₗ
    #    println(f, i)
    #  end
    #end
    #open("kappas.dat", "w") do f
    #  for i in κₗ
    #    println(f, i)
    #  end
    #end
    #print(ηₗ)
    
    # Raccolgo i coefficienti in due array (uno per quelli a sx, l'altro per
    # quelli a dx) per poterli disegnare assieme nei grafici.
    # (I coefficienti κ sono uno in meno degli Ω! Per ora pareggio le lunghezze
    # inserendo uno zero all'inizio dei κ…)
    osc_chain_coefficients_left = [Ωₗ [0; κₗ]]
    #osc_chain_coefficients_right = [Ωᵣ [0; κᵣ]]

    localcfs = [reverse(Ωₗ); repeat([ε], n_spin_sites)]
    interactioncfs = [reverse(κₗ); ηₗ; repeat([1], n_spin_sites-1)]
    hlist = twositeoperators(sites, localcfs, interactioncfs)
    #
    function links_odd(τ)
      return [(exp(-im * τ * h)) for h in hlist[1:2:end]]
    end
    function links_even(τ)
      return [(exp(-im * τ * h)) for h in hlist[2:2:end]]
    end
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             τ,
                             parameters["TS_expansion_order"])

    # Osservabili da misurare
    # =======================
    # - la corrente di spin
    list = spin_current_op_list(sites[range_spins])
    spin_current_ops = [embed_slice(sites, range_spins, j)
                        for j in list]
    # - l'occupazione degli autospazi dell'operatore numero
    projectors = [level_subspace_proj(sites[range_spins], n)
                  for n = 0:n_spin_sites]
    num_eigenspace_projs = [embed_slice(sites, range_spins, p)
                            for p in projectors]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto
    osc_sx_init_state = MPS(sites[range_osc_left], "0")
    spin_init_state = MPS(sites[range_spins], "X+")
    current_state = chain(osc_sx_init_state,
                          spin_init_state)

    # Misura della coerenza
    function spincoherence(ψ::MPS)
      # Lo spin si trova sull'ultimo sito del MPS
      orthogonalize!(ψ, length(ψ))
      A = last(ψ) * prime(last(ψ))
      # Contraggo gli indici di tipo Link e ottengo la matrice densità
      (α₁, α₂) = filter(i -> hastags(i, "Link"), inds(A))
      ρ = matrix(A * delta(α₁, α₂))
      return abs(tr(ρ * [0 1; 0 0]))
    end

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = [expect(current_state, "N")]
    bond_dimensions = [linkdims(current_state)]
    spin_current = [[real(inner(current_state, j * current_state))
                     for j in spin_current_ops]]
    spin_chain_levels = [levels(num_eigenspace_projs,
                                current_state)]
    coherence = Real[spincoherence(current_state)]

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
        push!(occ_n, expect(current_state, "N"))
        push!(coherence, spincoherence(current_state))
        push!(bond_dimensions, linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    snapshot = occ_n[end]

    ## Creo una tabella con i dati rilevanti da scrivere nel file di output
    #dict = Dict(:time => time_step_list[1:skip_steps:end])
    #tmp_list = hcat(occ_n...)
    #for (j, name) in enumerate([[Symbol("occ_n_left$n") for n∈n_osc_left:-1:1];
    #                            [Symbol("occ_n_spin$n") for n∈1:n_spin_sites];
    #                            [Symbol("occ_n_right$n") for n∈1:n_osc_right]])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(spin_current...)
    #for (j, name) in enumerate([Symbol("spin_current$n")
    #                            for n = 1:n_spin_sites-1])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(spin_chain_levels...)
    #for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_spin_sites])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(bond_dimensions...)
    #len = n_osc_left + n_spin_sites + n_osc_right
    #for (j, name) in enumerate([Symbol("bond_dim$n") for n ∈ 1:len-1])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #table = DataFrame(dict)
    #filename = replace(parameters["filename"], ".json" => "") * ".dat"
    ## Scrive la tabella su un file che ha la stessa estensione del file dei
    ## parametri, con estensione modificata.
    #CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    push!(coherence_super, coherence)
    push!(bond_dimensions_super, bond_dimensions)
    push!(osc_chain_coefficients_left_super, osc_chain_coefficients_left)
    push!(snapshot_super, snapshot)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  plot_size = (2, 0.5 + ceil(length(parameter_lists)/2)) .* (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

  # Grafico dei numeri di occupazione
  # ---------------------------------------------
  len = size(hcat(occ_n_super[begin]...), 1)
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=[string.(len-1:-1:1); "S"],
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione",
                         plot_size=plot_size)
  savefig(plt, "occ_n.png")

  # Grafico dell'elemento scelto della matrice densità ridotta dello spin
  # ---------------------------------------------------------------------
  plt = plot_time_series(coherence_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=[nothing],
                         linestyles=[:solid],
                         x_label=L"\lambda\, t",
                         y_label=L"(\rho_S)_{+,-}",
                         plot_title="Coerenza dello spin",
                         plot_size=plot_size
                        )
  savefig(plt, "coherence.png")


  # Grafico dei ranghi del MPS
  # --------------------------
  len = size(hcat(bond_dimensions_super[begin]...), 1)
  plt = plot_time_series(bond_dimensions_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=["($j,$(j+1))" for j=1:len],
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"\chi_{k,k+1}",
                         plot_title="Ranghi del MPS",
                         plot_size=plot_size
                        )
  savefig(plt, "bond_dimensions.png")

  #=
  # Grafico della corrente di spin
  # ------------------------------
  len = size(hcat(spin_current_super[begin]...), 1)
  plt = plot_time_series(spin_current_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=["($j,$(j+1))" for j=1:len],
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"j_{k,k+1}",
                         plot_title="Corrente di spin",
                         plot_size=plot_size
                        )
  savefig(plt, "spin_current.png")
 
  # Grafico dell'occupazione degli autospazi di N della catena di spin
  # ------------------------------------------------------------------
  # L'ultimo valore di ciascuna riga rappresenta la somma di tutti i
  # restanti valori.
  len = size(hcat(spin_chain_levels_super[begin]...), 1) - 1
  plt = plot_time_series(spin_chain_levels_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=[string.(0:len-1); "total"],
                         linestyles=[repeat([:solid], len); :dash],
                         x_label=L"\lambda\, t",
                         y_label=L"n",
                         plot_title="Occupazione degli autospazi "
                         * "della catena di spin",
                         plot_size=plot_size
                        )
  savefig(plt, "chain_levels.png")
  =#
 
  # Grafico dei coefficienti della chain map
  # ----------------------------------------
  plt = plot_standalone(osc_chain_coefficients_left_super,
                        parameter_lists;
                        labels=[L"\Omega_i", L"\kappa_i"],
                        linestyles=[:solid, :solid],
                        x_label=L"i",
                        y_label="Coefficiente",
                        plot_title="Coefficienti della catena di "*
                                   "oscillatori (sx)",
                        plot_size=plot_size
                        )
  savefig(plt, "osc_left_coefficients.png")

  # Istantanea dei numeri di occupazione alla fine
  # ----------------------------------------------
  plt = plot_standalone(snapshot_super,
                        parameter_lists;
                        labels=[nothing],
                        linestyles=[:solid],
                        x_label=L"i",
                        y_label="Numero di occupazione",
                        plot_title="Numeri di occupazione alla fine",
                        plot_size=plot_size
                        )
  savefig(plt, "snapshot.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
