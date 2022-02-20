#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV

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
  occ_n_mps_super = []
  bond_dimensions_super = []
  normalisation_super = []
  hermiticity_super = []

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
    γ = parameters["oscillator_damping_coefficient"]
    ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]
    oscdimL = parameters["left_oscillator_space_dimension"]
    oscdimR = parameters["right_oscillator_space_dimension"]
    n_spin_sites = 1

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    sites = [siteinds("HvOsc", 1; dim=oscdimL);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=oscdimR)]

    localcfs = [ω, ε, ω]
    interactioncfs = [κ, κ]
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ℓlist[begin] += (γ * op("Damping", sites[begin]; ω=ω, T=T) *
                     op("Id", sites[begin+1]))
    ℓlist[end] += (γ * op("Id", sites[end-1]) *
                   op("Damping", sites[end]; ω=ω, T=0))

    filename = parameters["filename"]
    println("-------------------- test ($filename) --------------------")
    # Definisco la parte unitaria dell'equazione di Lindblad
    # L₀(ρ) = -i[H,ρ]
    H12 = (ω * num(oscdimL) ⊗ id(2) ⊗ id(oscdimR) +
           κ * (a⁺(oscdimL)+a⁻(oscdimL)) ⊗ σˣ ⊗ id(oscdimR) +
           ε/2 * id(oscdimL) ⊗ σᶻ ⊗ id(oscdimR) +
           κ * id(oscdimL) ⊗ σˣ ⊗ (a⁺(oscdimR)+a⁻(oscdimR)) +
           ω * id(oscdimL) ⊗ id(2) ⊗ num(oscdimR))
    L₀(ρ) = -im * (H12 * ρ - ρ * H12)
    # e la parte D(ρ) di interazione con l'ambiente
    n = T != 0 ? (ℯ^(ω/T)-1)^(-1) : 0
    AL  = a⁻(oscdimL) ⊗ id(2) ⊗ id(oscdimR)
    AL⁺ = a⁺(oscdimL) ⊗ id(2) ⊗ id(oscdimR)
    AR  = id(oscdimL) ⊗ id(2) ⊗ a⁻(oscdimR)
    AR⁺ = id(oscdimL) ⊗ id(2) ⊗ a⁺(oscdimR)
    D(ρ) = (γ * (1+n) * (AL*ρ*AL⁺ - 0.5AL⁺*AL*ρ - 0.5ρ*AL⁺*AL) +
            γ * n * (AL⁺*ρ*AL - 0.5AL*AL⁺*ρ - 0.5*ρ*AL*AL⁺) +
            γ * (AR*ρ*AR⁺ - 0.5AR⁺*AR*ρ - 0.5ρ*AR⁺*AR))
    # Le unisco e calcolo la matrice rappresentativa nella base hermitiana.
    L(x) = L₀(x) + D(x)
    basis = [i⊗j⊗k for (i,j,k) in [Base.product(gellmannbasis(oscdimL), gellmannbasis(2), gellmannbasis(oscdimR))...]]
    manualL = vec(L, basis)
    maxothers = norm(vec(L₀, basis)[end,:], Inf)
    println("Nell'ultima riga della matrice che rappresenta la parte "*
            "unitaria dell'equazione di Lindblad, costruita manualmente, "*
            "le componenti sono tutte ≤ $maxothers in valore assoluto.")
    maxothers = norm(vec(D, basis)[end,:], Inf)
    println("Nell'ultima riga della matrice che rappresenta la parte "*
            "dissipativa dell'equazione di Lindblad, costruita manualmente, "*
            "le componenti sono tutte ≤ $maxothers in valore assoluto.")
    # Calcolo il suo esponenziale, che uso per calcolare la soluzione
    # esatta (numerica). Poi la confronto con l'operatore di evoluzione
    # ottenuto con ITensors.
    expL_manual = exp(time_step * manualL)
    # La matrice è troppo grossa per essere stampata: controllo perlomeno
    # che l'ultima riga sia tutta nulla.
    last = expL_manual[end,end]
    maxothers = norm(expL_manual[end,1:end-1], Inf)
    println("Nell'ultima riga della matrice di evoluzione temporale, "*
            "costruita manualmente:\n"*
            "· l'ultima componente è $last;\n"*
            "· gli altri elementi sono tutti ≤ $maxothers in valore assoluto.")

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
    # Ora cerco di combinare tutti i pezzi in `evo` in un unico tensore.
    # Assumo che ce ne siano 3, cioè di stare usando la decomposizione
    # di Trotter-Suzuki al 2° ordine.
    # Ogni u ∈ evo è un oggetto ITensor con due indici iₖ₁, iₖ₂ (che si
    # contrarrebbero con quelli dello stato MPS) e due indici iₖ₁', iₖ₂'.
    # Per moltiplicare tra loro questi tensori, devo lavorare un po' sugli
    # indici in modo da fare le giuste contrazioni.
    # Per calcolare u1 e u2, se non hanno siti in comune allora basta
    # fare u2*u1 e ottengo semplicemente il prodotto di Kronecker; se
    # invece hanno un sito k in comune, allora la moltiplicazione corretta
    # include la contrazione di iₖ' di u1 con iₖ di u2: devo quindi "primare"
    # la coppia (iₖ,iₖ') di u2 in modo che la contrazione sia giusta.
    # Dopo la moltiplicazione mi ritroverò con degli indici due volte primati,
    # che dovrò riportare allo stato iniziale.
    U = evo[1]
    for u ∈ evo[2:end]
      # Ottengo i tag degli indici in comune tra U e u.
      # Con questi tag individuo quali sono gli indici in u da "primare".
      primetaglist = unique(tags.(commoninds(U, u)))
      w = deepcopy(u)
      for t ∈ primetaglist
        prime!(w; tags=t)
      end
      U = w * U
      setprime!(U, 1; plev=2)
    end

    C = combiner(sites...)
    Cdag = combiner(prime.(sites)...)
    expL_mpo = matrix(U * Cdag * C)
    last = expL_mpo[end,end]
    maxothers = norm(expL_mpo[end,1:end-1], Inf)
    println("Nell'ultima riga della matrice di evoluzione temporale, "*
            "calcolata con ITensors:\n"*
            "· l'ultima componente è $last;\n"*
            "· gli altri elementi sono tutti ≤ $maxothers in valore assoluto.")

    diff = norm(expL_manual .- expL_mpo, 2)
    println("Differenza, in norma, tra la matrice di evoluzione temporale "*
            "calcolata manualmente e quella calcolata con ITensors: $diff")

    # Osservabili da misurare
    # =======================
    occ_n_list_mpo = [MPS(sites, ["vecN", "vecId", "vecId"]),
                      MPS(sites, ["vecId", "vecN", "vecId"]),
                      MPS(sites, ["vecId", "vecId", "vecN"])]
    #Nlist = [num(oscdimL) ⊗ id(2) ⊗ id(oscdimR),
    #         id(oscdimL) ⊗ [1 0; 0 0] ⊗ id(oscdimR),
    #         id(oscdimL) ⊗ id(2) ⊗ num(oscdimR),

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecId")

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # L'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    ρ_mps = chain(parse_init_state_osc(sites[1],
                    parameters["left_oscillator_initial_state"];
                    ω=ω, T=T),
                  MPS([state(sites[2], "Dn")]), 
                  MPS([state(sites[end], "0")]))

    #ρ_manual = vec(vector(currentstate * C), basis)

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n_mps = Vector{Real}[chop.([inner(s, ρ_mps) for s in occ_n_list_mpo])]
    bond_dimensions = Vector{Int}[linkdims(ρ_mps)]

    normalisation = Real[real(inner(full_trace, ρ_mps))]
    hermiticity = Real[0]

    # Evoluzione temporale
    # --------------------
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1
    for _ in time_step_list[2:end]
      ρ_mps = apply(evo,
                    ρ_mps,
                    cutoff=max_err,
                    maxdim=max_dim)
      #ρ_manual = expL_manual * ρ_manual
      if skip_count % skip_steps == 0
        #=
        Calcolo dapprima la traccia della matrice densità. Se non devia
        eccessivamente da 1, in ogni caso influisce sul valore delle
        osservabili che calcolo successivamente, che si modificano dello
        stesso fattore, e devono essere quindi corrette di un fattore pari
        al reciproco della traccia.
        =#
        trace = real(inner(full_trace, ρ_mps))

        push!(normalisation,
              trace)

        push!(occ_n_mps,
              [real(inner(s, ρ_mps)) for s in occ_n_list_mpo] ./ trace)

        push!(bond_dimensions,
              linkdims(ρ_mps))

        # Controllo che la matrice densità ridotta dell'oscillatore a sinistra
        # sia una valida matrice densità: hermitiana e semidefinita negativa.
        reduceddensitymat = partialtrace(sites, ρ_mps, 1)
        # Avverti solo se la matrice non è semidefinita positiva. Per calcolare
        # la positività degli autovalori devo tagliare via la loro parte reale,
        # praticamente assumendo che siano reali (cioè che mat sia hermitiana).
        for x in real.(eigvals(sum(reduceddensitymat .* gellmannbasis(oscdimL))))
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
    tmp_list = hcat(occ_n_mps...)
    for (j, name) in enumerate([:occ_n_left;
                              [Symbol("occ_n_spin$n") for n = 1:n_spin_sites];
                              :occ_n_right])
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
    push!(occ_n_mps_super, permutedims(hcat(occ_n_mps...)))
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
  N = size(occ_n_mps_super[begin])[2]
  plt = groupplot(timesteps_super,
                  occ_n_mps_super,
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
  occ_n_osc_left_super = [occ_n[:,1] for occ_n in occ_n_mps_super]
  plt = unifiedplot(timesteps_super,
                    occ_n_osc_left_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\langle n_L(t)\rangle",
                    plottitle="Occupazione dell'oscillatore sx",
                    plotsize=plotsize)

  savefig(plt, "occ_n_osc_left.png")

  #=
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
  plt = unifiedplot(timesteps_super,
                    normalisation_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\operatorname{tr}\,\rho(t)",
                    plottitle="Normalizzazione della matrice densità",
                    plotsize=plotsize)

  savefig(plt, "dm_normalisation.png")

  plt = unifiedplot(timesteps_super,
                    hermiticity_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\Vert\rho_\mathrm{L}(t)-\rho_\mathrm{L}(t)^\dagger\Vert",
                    plottitle="Controllo hermitianità della matrice densità",
                    plotsize=plotsize)

  savefig(plt, "hermiticity_monitor.png")

  #=
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
  =#

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
