using LinearAlgebra
using ITensors
using Combinatorics

# Spazio degli spin
# =================
ITensors.space(::SiteType"vecS=1/2") = 4

# Matrici di Pauli e affini
# -------------------------
σˣ = [0 1
      1 0]
σʸ = [0  -im
      im 0  ]
σᶻ = [1 0
      0 -1]
σ⁺ = [0 1
      0 0]
σ⁻ = [0 0
      1 0]
I₂ = [1 0
      0 1]

# Questi "stati" contengono sia degli stati veri e propri, come Up:Up e Dn:Dn,
# sia la vettorizzazione di alcuni operatori che mi servono per calcolare le
# osservabili sugli MPS (siccome sono vettori, devo definirli per forza così)
ITensors.state(::StateName"Up:Up", ::SiteType"vecS=1/2") = [ 1 0 0 0 ]
ITensors.state(::StateName"Dn:Dn", ::SiteType"vecS=1/2") = [ 0 0 0 1 ]
ITensors.state(::StateName"vecσx", ::SiteType"vecS=1/2") = [ 0 1 1 0 ]
ITensors.state(::StateName"vecσy", ::SiteType"vecS=1/2") = [ 0 im -im 0 ]
ITensors.state(::StateName"vecσz", ::SiteType"vecS=1/2") = [ 1 0 0 -1 ]
ITensors.state(::StateName"vecid", ::SiteType"vecS=1/2") = [ 1 0 0 1 ]
ITensors.state(::StateName"zero", ::SiteType"vecS=1/2")  = [ 0 0 0 0 ]

# Operatori semplici sullo spazio degli spin (vettorizzato)
# ---------------------------------------------------------
# - identità
ITensors.op(::OpName"id:id", ::SiteType"vecS=1/2") = kron(I₂, I₂)
# - termini per l'Hamiltoniano locale
ITensors.op(::OpName"σz:id", ::SiteType"vecS=1/2") = kron(σᶻ, I₂)
ITensors.op(::OpName"id:σz", ::SiteType"vecS=1/2") = kron(I₂, σᶻ)
# - termini per l'Hamiltoniano bilocale
ITensors.op(::OpName"id:σ+", ::SiteType"vecS=1/2") = kron(I₂, σ⁺)
ITensors.op(::OpName"σ+:id", ::SiteType"vecS=1/2") = kron(σ⁺, I₂)
ITensors.op(::OpName"id:σ-", ::SiteType"vecS=1/2") = kron(I₂, σ⁻)
ITensors.op(::OpName"σ-:id", ::SiteType"vecS=1/2") = kron(σ⁻, I₂)
# - termini di smorzamento
ITensors.op(::OpName"σx:σx", ::SiteType"vecS=1/2") = kron(σˣ, σˣ)
ITensors.op(::OpName"σx:id", ::SiteType"vecS=1/2") = kron(σˣ, I₂)
ITensors.op(::OpName"id:σx", ::SiteType"vecS=1/2") = kron(I₂, σˣ)

# Composizione dell'Hamiltoniano per i termini di spin
# ----------------------------------------------------
# - termini locali dell'Hamiltoniano
function ITensors.op(::OpName"H1loc", ::SiteType"vecS=1/2", s::Index)
  h = op("σz:id", s) - op("id:σz", s)
  return 0.5im * h
end
# - termini bilocali dell'Hamiltoniano
function ITensors.op(::OpName"HspinInt", ::SiteType"vecS=1/2", s1::Index, s2::Index)
  h = op("id:σ-", s1) * op("id:σ+", s2) +
      op("id:σ+", s1) * op("id:σ-", s2) -
      op("σ-:id", s1) * op("σ+:id", s2) -
      op("σ+:id", s1) * op("σ-:id", s2)
  return 0.5im * h
end
# - esponenziale del termine che coinvolge solo spin
function ITensors.op(::OpName"expHspin", ::SiteType"vecS=1/2", s1::Index, s2::Index; t::Number, ε::Number)
  ℓ = 0.5ε * op("H1loc", s1) * op("id:id", s2) +
      0.5ε * op("id:id", s1) * op("H1loc", s2) +
      op("HspinInt", s1, s2)
  return exp(t * ℓ)
end
# - termine di smorzamento per uno spin
function ITensors.op(::OpName"damping", ::SiteType"vecS=1/2", s::Index)
  d = op("σx:σx", s) - op("id:id", s)
  return d
end

# Base di autostati per la catena
# -------------------------------
#= Come misurare il "contributo" di ogni autospazio dell'operatore numero
   N su uno stato rappresentato dalla matrice densità ρ?
   Se Pₙ è la proiezione sull'autospazio dell'autovalore n di N, il
   valore cercato è cₙ = tr(ρ Pₙ).
   Il proiettore Pₙ lo posso costruire sommando i proiettori su ogni singolo
   autostato nell'n-esimo autospazio: è un metodo grezzo ma dovrebbe
   funzionare. Calcolando prima dell'avvio dell'evoluzione temporale tutti
   i proiettori si dovrebbe risparmiare tempo, dovento effettuare poi solo
   n_sites operazioni ad ogni istante di tempo.                         
=#
# La seguente funzione restituisce i nomi per costruire i MPS degli
# stati della base dell'intero spazio della catena (suddivisi per
# numero di occupazione complessivo).
function chain_basis_states(n_sites::Int, level::Int)
  return unique(permutations(vcat(repeat(["Up:Up"], level),
                                  repeat(["Dn:Dn"], n_sites - level))))
end

# La seguente funzione crea un proiettore su ciascun sottospazio con
# livello di occupazione definito.
function level_subspace_proj(sites::Vector{Index{Int64}}, level::Int)
  # Metodo "crudo": prendo tutti i vettori della base ortonormale di
  # ciascun autospazio, ne creo i proiettori ortogonali e li sommo tutti.
  # Forse poco efficiente, ma funziona.
  N = length(sites)
  proj = MPS(sites, "zero")
  for names in chain_basis_states(N, level)
    proj = add(proj, MPS(sites, names); cutoff=1e-10)
  end
  return proj
end
