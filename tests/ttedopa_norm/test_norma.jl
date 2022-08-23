#!/usr/bin/julia

using LinearAlgebra, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using ITensors, PseudomodesTTEDOPA

disablegrifqtech()

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let  
const ⊗ = kron

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

    a⁻(dim::Int)  = diagm(1  => sqrt.(1:dim-1))
    a⁺(dim::Int)  = diagm(-1 => sqrt.(1:dim-1))
    num(dim::Int) = diagm(0 =>  sqrt.(0:dim-1))
    id(dim::Int)  = Matrix{Int}(I, dim, dim)
    σˣ          = [0 1; 1  0]
    σᶻ          = [1 0; 0 -1]

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
    manualL = PseudomodesTTEDOPA.vec(L, basis)
    maxothers = norm(PseudomodesTTEDOPA.vec(L₀, basis)[end,:], Inf)
    println("Nell'ultima riga della matrice che rappresenta la parte "*
            "unitaria dell'equazione di Lindblad, costruita manualmente, "*
            "le componenti sono tutte ≤ $maxothers in valore assoluto.")
    maxothers = norm(PseudomodesTTEDOPA.vec(D, basis)[end,:], Inf)
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
    occ_n_list_mpo = [MPS(sites,
                          [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                      for n ∈ 1:length(sites)]
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

  # Plots
  # -----
  @info "Drawing plots."

  # Common options for group plots
  nrows = Int(ceil(tot_sim_n / 2))
  common_opts = @pgf {
    no_markers,
    grid       = "major",
    legend_pos = "outer north east",
    "every axis plot/.append style" = "thick"
  }
  group_opts = @pgf {
    group_style = {
      group_size        = "$nrows by 2",
      y_descriptions_at = "edge left",
      horizontal_sep    = "2cm",
      vertical_sep      = "2cm"
    },
    common_opts...
  }

  # Occupation numbers, left chain
  occn_left = [mat[:, range]
               for (mat, range) ∈ zip(occn_super, range_osc_left_super)]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           occ_n_mps_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["L"; string.(1:N-2); "R"] ))
      push!(grp, ax)
    end
    pgfsave("population.pdf", grp)
  end

  # Population, left oscillator
  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\langla n_L(t)\rangle",
               title = "Population of the left pseudomode",
               common_opts...
              })
    for (t, y, p, col) ∈ zip(timesteps_super,
                             [occ_n[:,1] for occ_n ∈ occ_n_mps_super],
                             parameter_lists,
                             distinguishable_colors(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("population_left_pseudomode.pdf", ax)
  end

  # Bond dimensions
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\chi_{i,i+1}(t)",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           bond_dimensions_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2) - 1
      sitelabels = ["L"; string.("S", 1:N); "R"]
      for (y, c, ls) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( consecutivepairs(sitelabels) ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions.pdf", grp)
  end

  # Trace of the density matrix
  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\mathrm{tr}\rho(t)",
               title = "Normalisation",
               common_opts...
              })
    for (t, y, p, col) ∈ zip(timesteps_super,
                             normalisation_super,
                             parameter_lists,
                             readablecolours(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("normalisation.pdf", ax)
  end

  # Hermiticity of the reduced density matrix of the left pseudomode
  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\Vert\rho_\mathrm{L}(t)-\rho_\mathrm{L}(t)^\dagger\Vert",
               title = "Hermiticity of red. density matrix of left pseudomode",
               common_opts...
              })
    for (t, y, p, col) ∈ zip(timesteps_super,
                             hermiticity_super,
                             parameter_lists,
                             distinguishable_colors(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("hermiticity.pdf", ax)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
