#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV
using DelimitedFiles

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
  timesteps_super = []
  occ_n_super = []
  spin_current_super = []
  bond_dimensions_super = []
  chain_levels_super = []
  osc_levels_left_super = []
  osc_levels_right_super = []
  normalisation_super = []
  hermiticity_super = []

  # Precaricamento
  # ==============
  # Se in tutte le liste di parametri il numero di siti è lo stesso, posso
  # definire qui una volta per tutte alcuni elementi "pesanti" che servono dopo.
  n_spin_sites_list = [p["number_of_spin_sites"] for p in parameter_lists]
  osc_dim_list = [p["oscillator_space_dimension"] for p in parameter_lists]
  if allequal(n_spin_sites_list) && allequal(osc_dim_list)
    preload = true
    n_spin_sites = first(n_spin_sites_list)
    osc_dim = first(osc_dim_list)

    spin_range = 1 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 1; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=osc_dim)]

    single_ex_states = [embed_slice(sites,
                                    spin_range,
                                    single_ex_state(sites[spin_range], k))
                        for k = 1:n_spin_sites]
    # - i numeri di occupazione: per gli spin della catena si prende il prodotto
    #   interno con gli elementi di single_ex_states già definiti; per gli
    #   oscillatori, invece, uso
    osc_num_sx = MPS(sites, ["vecN"; repeat(["vecId"], n_spin_sites+1)])
    osc_num_dx = MPS(sites, [repeat(["vecId"], n_spin_sites+1); "vecN"])

    occ_n_list = [osc_num_sx; single_ex_states; osc_num_dx]

    # - la corrente di spin
    # Prima costruisco gli operatori sulla catena di spin, poi li
    # estendo con l'identità sui restanti siti.
    spin_current_ops = [embed_slice(sites, spin_range, j)
                        for j in spin_current_op_list(sites[spin_range])]

    # - l'occupazione degli autospazi dell'operatore numero
    # Ad ogni istante proietto lo stato corrente sugli autostati
    # dell'operatore numero della catena di spin, vale a dire calcolo
    # tr(ρₛ Pₙ) dove ρₛ è la matrice densità ridotta della catena di spin
    # e Pₙ è il proiettore ortogonale sull'n-esimo autospazio di N
    num_eigenspace_projs = [embed_slice(sites,
                                        spin_range,
                                        level_subspace_proj(sites[spin_range], n))
                            for n=0:n_spin_sites]

    # - l'occupazione dei livelli degli oscillatori
    osc_levels_projs_left = [embed_slice(sites,
                                         1:1,
                                         osc_levels_proj(sites[1], n))
                             for n=0:osc_dim-1]
    osc_levels_projs_right = [embed_slice(sites,
                                          n_spin_sites+2:n_spin_sites+2,
                                          osc_levels_proj(sites[end], n))
                              for n=0:osc_dim-1]

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecId")
  else
    preload = false
  end

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

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
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    if !preload
      n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
      spin_range = 1 .+ (1:n_spin_sites)

      sites = [siteinds("HvOsc", 1; dim=osc_dim);
               siteinds("HvS=1/2", n_spin_sites);
               siteinds("HvOsc", 1; dim=osc_dim)]

      single_ex_states = [chain(MPS(sites[1:1], "vecId"),
                                single_ex_state(sites[2:end-1], k),
                                MPS(sites[end:end], "vecId"))
                          for k = 1:n_spin_sites]
    end

    #= Definizione degli operatori nell'equazione di Lindblad
       ======================================================
       I siti del sistema sono numerati come segue:
       | 1 | 2 | ... | n_spin_sites | n_spin_sites+1 | n_spin_sites+2 |
         ↑   │                        │          ↑
         │   └───────────┬────────────┘          │
         │               │                       │
         │        catena di spin                 │
       oscillatore sx                    oscillatore dx
    =#
    localcfs = [ω; repeat([ε], n_spin_sites); ω]
    interactioncfs = [κ; repeat([1], n_spin_sites-1); κ]
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ℓlist[begin] += γₗ * op("Damping", sites[begin]; ω=ω, T=T) *
                         op("Id", sites[begin+1])
    ℓlist[end] += γᵣ * op("Id", sites[end-1]) *
                       op("Damping", sites[end]; ω=ω, T=0)

    println("--------------------------- test ---------------------------")
    # Definisco la parte unitaria dell'equazione di Lindblad
    # L0(ρ) = -i[H,ρ]
    A = a⁻(osc_dim)
    A⁺ = a⁺(osc_dim)
    H12 = ω * num(osc_dim) ⊗ id(2) + ε/4 * id(osc_dim) ⊗ σᶻ + κ * (A⁺+A) ⊗ σˣ
    L0(ρ) = -im * (H12 * ρ - ρ * H12)
    # e la parte D(ρ) di interazione con l'ambiente
    n = T != 0 ? (ℯ^(ω/T)-1)^(-1) : 0
    A = a⁻(osc_dim) ⊗ id(2)
    A⁺ = a⁺(osc_dim) ⊗ id(2)
    D(ρ) = γₗ * (1+n) * (A*ρ*A⁺ - 0.5A⁺*A*ρ - 0.5ρ*A⁺*A) + γₗ * n * (A⁺*ρ*A - 0.5A*A⁺*ρ - 0.5*ρ*A*A⁺)
    # Le unisco e calcolo la matrice rappresentativa nella base hermitiana.
    f(x) = L0(x) + D(x)
    basis = [i⊗j for (i,j) in [Base.product(gellmannbasis(osc_dim), gellmannbasis(2))...]]
    Ma = vec(f, basis)
    # Calcolo qui invece la stessa matrice, ma costruita con gli oggetti
    # di ITensors (che è quella che verrà usata nella simulazione.
    C = combiner(sites[begin], sites[begin+1])
    Cdag = combiner(sites[begin]', sites[begin+1]')
    Mb = matrix(ℓlist[1] * Cdag * C)
    # Stampo la matrice usata da ITensors per vedere se ha le giuste
    # caratteristiche. Arrotondo un po' i numeri per salvare spazio.
    println("matrice rapp. dell'operatore L(ρ) nella base hermitiana, "*
            "costruita con ITensors:")
    writedlm(stdout, round.(Mb; digits=3))
    println("------------------------------------------------------------")
    println("differenza tra la matrice di ITensors e quella costruita "*
            "a mano:")
    # Stampo la differenza tra le due matrici (troncando i numeri
    # piccolissimi)
    writedlm(stdout, chop.(Ma .- Mb))
    println("------------------------------------------------------------")
    # e anche la norma della differenza, così per matrici enormi si vede
    # meglio quanto differiscono
    diff = norm(Ma .- Mb, 2)
    println("differenza in norma (p=2): $diff")
    println("------------------------------------------------------------")

    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             time_step,
                             parameters["TS_expansion_order"])

    # Osservabili da misurare
    # =======================
    if !preload
      # - i numeri di occupazione
      osc_num_sx = MPS(sites, ["vecN"; repeat(["vecId"], n_spin_sites+1)])
      osc_num_dx = MPS(sites, [repeat(["vecId"], n_spin_sites+1); "vecN"])
      occ_n_list = [osc_num_sx; single_ex_states; osc_num_dx]

      # - la corrente di spin
      spin_current_ops = [embed_slice(sites, spin_range, j)
                          for j in spin_current_op_list(sites[spin_range])]

      # - l'occupazione degli autospazi dell'operatore numero
      num_eigenspace_projs = [embed_slice(sites,
                                          spin_range,
                                          level_subspace_proj(sites[spin_range], n))
                              for n=0:n_spin_sites]

      # - l'occupazione dei livelli degli oscillatori
      osc_levels_projs_left = [embed_slice(sites,
                                           1:1,
                                           osc_levels_proj(sites[1], n))
                               for n=0:osc_dim-1]
      osc_levels_projs_right = [embed_slice(sites,
                                            n_spin_sites+2:n_spin_sites+2,
                                            osc_levels_proj(sites[end], n))
                                for n=0:osc_dim-1]

      # - la normalizzazione (cioè la traccia) della matrice densità
      full_trace = MPS(sites, "vecId")
    end

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # L'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    current_state = chain(parse_init_state_osc(sites[1],
                                 parameters["left_oscillator_initial_state"];
                                 ω=ω, T=T),
                          parse_init_state(sites[2:end-1],
                                           parameters["chain_initial_state"]),
                          parse_init_state_osc(sites[end], "empty"))

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = Vector{Real}[chop.([inner(s, current_state) for s in occ_n_list])]
    bond_dimensions = Vector{Int}[linkdims(current_state)]
    spin_current = Vector{Real}[real.([inner(j, current_state)
                                       for j in spin_current_ops])]
    chain_levels = Vector{Real}[real.(levels(num_eigenspace_projs,
                                             current_state))]
    osc_levels_left = Vector{Real}[real.(levels(osc_levels_projs_left,
                                                current_state))]
    osc_levels_right = Vector{Real}[real.(levels(osc_levels_projs_right,
                                                 current_state))]

    normalisation = Real[real(inner(full_trace, current_state))]
    hermiticity = Real[0]

    # Evoluzione temporale
    # --------------------
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state,
                            cutoff=max_err,
                            maxdim=max_dim)
      if skip_count % skip_steps == 0
        #=
        Calcolo dapprima la traccia della matrice densità. Se non devia
        eccessivamente da 1, in ogni caso influisce sul valore delle
        osservabili che calcolo successivamente, che si modificano dello
        stesso fattore, e devono essere quindi corrette di un fattore pari
        al reciproco della traccia.
        =#
        trace = real(inner(full_trace, current_state))

        push!(normalisation,
              trace)

        push!(occ_n,
              [real(inner(s, current_state)) for s in occ_n_list] ./ trace)

        push!(spin_current,
              [real(inner(j, current_state)) for j in spin_current_ops] ./ trace)
        push!(chain_levels,
              levels(num_eigenspace_projs, current_state) ./ trace)
        push!(osc_levels_left,
              levels(osc_levels_projs_left, current_state) ./ trace)
        push!(osc_levels_right,
              levels(osc_levels_projs_right, current_state) ./ trace)
        push!(bond_dimensions,
              linkdims(current_state))

        # Controllo che la matrice densità ridotta dell'oscillatore a sinistra
        # sia una valida matrice densità: hermitiana e semidefinita negativa.
        reduceddensitymat = partialtrace(sites, current_state, 1)
        # Avverti solo se la matrice non è semidefinita positiva. Per calcolare
        # la positività degli autovalori devo tagliare via la loro parte reale,
        # praticamente assumendo che siano reali (cioè che mat sia hermitiana).
        for x in real.(eigvals(sum(reduceddensitymat .* gellmannbasis(osc_dim))))
          if x < -max_err
            @warn "La matrice densità del primo sito non è semidefinita positiva: trovato $x"
          end
        end
        # Siccome la base di Gell-Mann è hermitiana, `reduceddensitymat` sarà
        # hermitiana se tutti i coefficienti sono reali. Calcolo quindi la
        # deviazione dal risultato ideale come la somma dei valori assoluti
        # della parte immaginaria dei coefficienti del vettore.
        push!(hermiticity,
              sum(abs.(imag.(reduceddensitymat))))
      end
      next!(progress)
      skip_count += 1
    end

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => time_step_list[1:skip_steps:end])
    tmp_list = hcat(occ_n...)
    for (j, name) in enumerate([:occ_n_left;
                              [Symbol("occ_n_spin$n") for n = 1:n_spin_sites];
                              :occ_n_right])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(spin_current...)
    for (j, name) in enumerate([Symbol("spin_current$n") for n = 1:n_spin_sites-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(osc_levels_left...)
    for (j, name) in enumerate([Symbol("levels_left$n") for n = 0:osc_dim-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(chain_levels...)
    for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_spin_sites])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(osc_levels_right...)
    for (j, name) in enumerate([Symbol("levels_right$n") for n = 0:osc_dim:-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(bond_dimensions...)
    len = n_spin_sites + 2
    for (j, name) in enumerate([Symbol("bond_dim$n")
                                for n ∈ 1:len-1])
      push!(dict, name => tmp_list[j,:])
    end
    push!(dict, :full_trace => normalisation)
    push!(dict, :hermiticity => hermiticity)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list[1:skip_steps:end])
    push!(occ_n_super, permutedims(hcat(occ_n...)))
    push!(spin_current_super, permutedims(hcat(spin_current...)))
    push!(chain_levels_super, permutedims(hcat(chain_levels...)))
    push!(osc_levels_left_super, permutedims(hcat(osc_levels_left...)))
    push!(osc_levels_right_super, permutedims(hcat(osc_levels_right...)))
    push!(bond_dimensions_super, permutedims(hcat(bond_dimensions...)))
    push!(normalisation_super, normalisation)
    push!(hermiticity_super, hermiticity)
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

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  N = size(occ_n_super[begin])[2]
  plt = groupplot(timesteps_super,
                  occ_n_super,
                  parameter_lists;
                  labels=["L" string.(1:N-2)... "R"],
                  linestyles=[:dash repeat([:solid], N-2)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione",
                  plotsize=plotsize)

  savefig(plt, "occ_n_all.png")

  # Grafico dell'occupazione del primo oscillatore (riunito)
  # -------------------------------------------------------
  # Estraggo da occ_n_super i valori dell'oscillatore sinistro.
  occ_n_osc_left_super = [occ_n[:,1] for occ_n in occ_n_super]
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
  spinsonly = [mat[:, 2:end-1] for mat in occ_n_super]
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
  plt = unifiedplot(timesteps_super,
                    normalisation_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\operatorname{tr}\,\rho(t)",
                    plottitle="Normalizzazione della matrice densità",
                    plotsize=plotsize)

  savefig(plt, "dm_normalisation.png")

  # Grafico della traccia della matrice densità
  # -------------------------------------------
  # Questo serve più che altro per controllare che rimanga sempre pari a 1.
  plt = unifiedplot(timesteps_super,
                    hermiticity_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\Vert\rho_\mathrm{L}(t)-\rho_\mathrm{L}(t)^\dagger\Vert",
                    plottitle="Controllo hermitianità della matrice densità",
                    plotsize=plotsize)

  savefig(plt, "hermiticity_monitor.png")

  # Grafico della corrente di spin
  # ------------------------------
  N = size(spin_current_super[begin])[2]
  plt = groupplot(timesteps_super,
                  spin_current_super,
                  parameter_lists;
                  labels=hcat(["($j,$(j+1))" for j ∈ 1:N]...),
                  linestyles=hcat(repeat([:solid], N)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"j_{k,k+1}(t)",
                  plottitle="Corrente di spin",
                  plotsize=plotsize)

  savefig(plt, "spin_current.png")

  # Grafico dell'occupazione degli autospazi di N della catena di spin
  # ------------------------------------------------------------------
  # L'ultimo valore di ciascuna riga rappresenta la somma di tutti i
  # restanti valori.
  N = size(chain_levels_super[begin])[2] - 1
  plt = groupplot(timesteps_super,
                  chain_levels_super,
                  parameter_lists;
                  labels=[string.(0:N-1)... "total"],
                  linestyles=[repeat([:solid], N)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"n(",
                  plottitle="Occupazione degli autospazi "
                  * "della catena di spin",
                  plotsize=plotsize)

  savefig(plt, "chain_levels.png")

  # Grafico dell'occupazione dei livelli degli oscillatori
  # ------------------------------------------------------
  for (list, pos) in zip([osc_levels_left_super, osc_levels_right_super],
                         ["sx", "dx"])
    N = size(list[begin])[2] - 1
    plt = groupplot(timesteps_super,
                    list,
                    parameter_lists;
                    labels=[string.(0:N-1)... "total"],
                    linestyles=[repeat([:solid], N)... :dash],
                    commonxlabel=L"\lambda\, t",
                    commonylabel=L"n",
                    plottitle="Occupazione degli autospazi "
                    * "dell'oscillatore $pos",
                    plotsize=plotsize)

    savefig(plt, "osc_levels_$pos.png")
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
