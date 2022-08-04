using PolyChaos
using QuadGK

"""
    getchaincoefficients(envparameters)

Return the Ω, η, κ coefficients of the T-TEDOPA chain, from the information given
in the `envparameters` dictionary.
"""
function getchaincoefficients(envparameters)
  n_osc = envparameters["number_of_oscillators"]
  #=
  Parametri delle catene di oscillatori
  Ci sono due casi qui, in base a cosa viene specificato nel JSON:
  1) viene indicato un file alla voce "X_chain_coefficients_file"
     dove X è "left" o "right". In questo caso, prendo i
     coefficienti da quel file.
  2) viene descritta una densità spettrale, con i suoi parametri,
     alle voci "spectral_density_parameters" e "…_function". In
     questo caso, calcolo i coefficienti con le funzioni del
     T-TEDOPA come sempre.
  In ogni caso, la temperatura dell'ambiente è completamente
  inserita nei coefficienti, così come la frequenza di taglio,
  quindi non verranno specificati.
  È ammessa pure una situazione mista, in cui una catena è data
  tramite i coefficienti e l'altra tramite una funzione.
  Ciò che non è ammesso è dare contemporaneamente funzione e
  coefficienti per il medesimo ambiente: qui lancio un errore.
  =#
  if (haskey(envparameters, "chain_coefficients_file") &&
      !haskey(envparameters, "spectral_density_function"))
    ccoeffs = DataFrame(CSV.File(envparameters["chain_coefficients_file"]))
    if size(ccoeffs, 1) ≥ n_osc
      # TODO: studiare il caso particolare n_osc == 1
      # Se è >, allora i coefficienti in eccesso vengono ignorati;
      # se è =, allora tutto combacia e tutti i coefficienti sono
      # considerati.
      Ω = ccoeffs[1:n_osc, :loc]
      η = ccoeffs[1,       :int]
      κ = ccoeffs[2:n_osc, :int]
    else
      error("File $(envparameters["chain_coefficients_file"]) does "*
            "not contain enough coefficients.")
    end
  elseif (!haskey(envparameters, "chain_coefficients_file") &&
          haskey(envparameters, "spectral_density_function"))
    # Codice per creare la funzione a partire dalla stringa tratto da
    # https://stackoverflow.com/a/53134127/4160978 
    fn = envparameters["spectral_density_function"]
    tmp = eval(Meta.parse("(a, x) -> " * fn))
    sdf = x -> Base.invokelatest(tmp,
                                 envparameters["spectral_density_parameters"], x)

    ωc = envparameters["frequency_cutoff"]
    T = envparameters["temperature"]

    if T == 0
      Ω, κ, η = chainmapcoefficients(sdf,
                                          (0, ωc),
                                          n_osc-1;
                                          Nquad=envparameters["PolyChaos_nquad"],
                                          discretization=lanczos)
    else
      sdf_thermalised = ω -> thermalisedJ(sdf, ω, T)
      Ω, κ, η = chainmapcoefficients(sdf_thermalised,
                                          (-ωc, 0, ωc),
                                          n_osc-1;
                                          Nquad=envparameters["PolyChaos_nquad"],
                                          discretization=lanczos)
    end
  end
  return Ω, κ, η
end

"""
    thermalisedJ(J::Function, ω::Real, T::Real)

Create a thermalised spectral function from `J`, at the temperature
`T`, then return its value at `ω`.
"""
function thermalisedJ(J::Function,
                      ω::Real,
                      T::Real)
  # Per evitare divisioni per zero, scrivo a parte il fattore:
  # lim T->0 coth(x/T)=Θ(x).
  # Si potrebbe anche scrivere f=1 e basta, assumendo che chi chiama
  # la funzione abbia usato un supporto che escluda già (-∞,0), ma
  # così almeno sto sicuro.
  if T == 0
    f = ω > 0 ? one(ω) : zero(ω)
  else
    f = 0.5(1 + coth(0.5 * ω / T))
  end
  return f * sign(ω) * J(abs(ω))
end

"""
    chainmapcoefficients(J::Function, support, L::Int; <keyword arguments>)

Compute the first `L` coefficients of the chain-map Hamiltonian obtained from
the spectral density `J` defined over `support`.

# Arguments
- `J::Function`: non-negative function defined (at least) on `support`.
- `support`: support of `J`; it can be a pair of real numbers, denoting an
  interval, but can be more generally a list of increasing real numbers `[a1, 
  a2, a3, …, aN]``, that will be interpreted as ``(a1,a2) ∪ (a2,a3) ∪ ⋯ ∪ 
  (aN-1,aN)``.
  The subdivision is ignored for the calculation of the chain coefficients, but
  it can become useful to compute the integral of `J` over `support`, since the
  numerical integration algorithm may need to exclude some intermediate points
  from the domain to work (e.g. if they are singularities).
- `L::Int`: number of oscillators in the chain.
- `Nquad::Int`: a parameter passed to OrthoPoly, which determines the number of
  nodes used for the quadrature method when numerical integration is performed.

Keyword arguments are passed on to the OrthoPoly constructor of the set of
orthogonal polynomials.

It returns the tuple `(Ω,κ,η)` containing
- the single-site energies `Ω`,
- the coupling coefficients `κ` between oscillators,
- the coupling coefficient `η` between the first oscillator and the system.

The spectral function `J` is associated to the recursion coefficients ``αₙ`` and
``βₙ`` which make up the recursion formula ``x πₙ(x) = πₙ₊₁(x) + αₙ πₙ(x) +
βₙ πₙ₋₁(x)`` for the monic orthogonal polynomials ``{πₙ}ₙ``, ``n ∈ ℕ``
determined by `J`. In the formula, π₋₁ is the null polynomial.
The chain coefficients are then given by
- ``Ωᵢ = αᵢ``, with ``i ∈ {1,…,L}``,
- ``κᵢ = sqrt(βᵢ₊₁)``, with ``i ∈ {1,…,L-1}``,
while `η` is the integral of `J` over its support.
"""
function chainmapcoefficients(J::Function, support, L::Int; kwargs...)
  measure = PolyChaos.Measure("measure", J, (support[begin], support[end]), false, Dict())
  poly = PolyChaos.OrthoPoly("poly", L, measure; kwargs...)
  #=
  Per costruire una serie di L oscillatori, servono i coefficienti αᵢ e βᵢ da
  i=0 a i=L-1: siccome però gli array sono indicizzati partendo da 1 in Julia,
  mi serviranno α[1:L] e β[1:L].
  A partire da questi coefficienti, indicizzando i siti degli oscillatori da 1
  a L, le frequenze Ωᵢ di ciascun oscillatore e la costante di interazione κᵢ
  tra gli oscillatori (i,i+1) sono date da
  Ωᵢ = αᵢ            per i∈{1,…,L},
  κᵢ = sqrt(βᵢ₊₁)    per i∈{1,…,L-1}.
  Il coefficiente β₀ rimane inutilizzato (onestamente, siccome a livello
  teorico è arbitrario, il suo valore dipende dall'implementazione di
  PolyChaos, quindi non so neanche quanto vale).
  Rimane il coefficiente η di interazione tra oscillatore e spin, che calcolo
  come l'integrale di J sul supporto dato.
  =#
  α = coeffs(poly)[:,1]
  β = coeffs(poly)[:,2]
  Ω = α
  κ = sqrt.(β[2:end])
  η = sqrt(quadgk(J, support...)[begin])
  return Ω, κ, η
end
