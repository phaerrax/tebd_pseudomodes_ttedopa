using LinearAlgebra
using ITensors
using Combinatorics

const ⊗ = kron

# Matrici di Pauli e affini
# -------------------------
σˣ = [0 1; 1 0]
σʸ = [0 -im; im 0]
σᶻ = [1 0; 0 -1]
σ⁺ = [0 1; 0 0]
σ⁻ = [0 0; 1 0]
I₂ = [1 0; 0 1]
ê₊ = [1; 0]
ê₋ = [0; 1]

# Il vettore nullo:
ITensors.state(::StateName"0", ::SiteType"S=1/2") = [0; 0]
# Definisco l'operatore numero per un singolo spin, che altro non è che
# la proiezione sullo stato |↑⟩.
ITensors.op(::OpName"N", s::SiteType"S=1/2") = op(OpName"ProjUp"(), s)
# La matrice identità
ITensors.op(::OpName"Id", ::SiteType"S=1/2") = I₂
# La matrice nulla
ITensors.op(::OpName"0", ::SiteType"S=1/2") = [0 0; 0 0]
# Matrici di Pauli
ITensors.op(::OpName"σx", ::SiteType"S=1/2") = σˣ
ITensors.op(::OpName"σy", ::SiteType"S=1/2") = σʸ
ITensors.op(::OpName"σz", ::SiteType"S=1/2") = σᶻ
# Operatori di scala
ITensors.op(::OpName"σ+", ::SiteType"S=1/2") = σ⁺
ITensors.op(::OpName"σ-", ::SiteType"S=1/2") = σ⁻

# Hamiltoniano locale di una coppia di spin adiacenti
function ITensors.op(::OpName"SpinLoc",
					 ::SiteType"S=1/2",
					 s1::Index,
					 s2::Index;
					 ε1::Real=1,
					 ε2::Real=1)
  return 0.5ε1 * op("σz", s1) * op("Id", s2) +
         0.5ε2 * op("Id", s1) * op("σz", s2)
end
# Hamiltoniano di interazione tra due spin
function ITensors.op(::OpName"SpinInt",
                     ::SiteType"S=1/2",
                     s1::Index,
                     s2::Index;
                     λ::Real=1)
  return -0.5λ * (op("σ-", s1) * op("σ+", s2) +
                  op("σ+", s1) * op("σ-", s2))
end

# Spazio degli spin vettorizzato
# ==============================
ITensors.space(::SiteType"vecS=1/2") = 4

# Stati (veri e propri)
# ---------------------
ITensors.state(::StateName"Up", ::SiteType"vecS=1/2") = ê₊ ⊗ ê₊
ITensors.state(::StateName"Dn", ::SiteType"vecS=1/2") = ê₋ ⊗ ê₋

# Stati che sono operatori vettorizzati (per costruire le osservabili)
# --------------------------------------------------------------------
ITensors.state(::StateName"vecσx", ::SiteType"vecS=1/2") = vcat(σˣ[:])
ITensors.state(::StateName"vecσy", ::SiteType"vecS=1/2") = vcat(σʸ[:])
ITensors.state(::StateName"vecσz", ::SiteType"vecS=1/2") = vcat(σᶻ[:])
ITensors.state(::StateName"vecId", ::SiteType"vecS=1/2") = vcat(I₂[:])
ITensors.state(::StateName"vec0", ::SiteType"vecS=1/2") = [0; 0; 0; 0]

# Operatori
# ---------
# - identità
ITensors.op(::OpName"Id:Id", ::SiteType"vecS=1/2") = I₂ ⊗ I₂
# - termini per l'Hamiltoniano locale
ITensors.op(::OpName"σz:Id", ::SiteType"vecS=1/2") = σᶻ ⊗ I₂
ITensors.op(::OpName"Id:σz", ::SiteType"vecS=1/2") = I₂ ⊗ σᶻ
# - termini per l'Hamiltoniano bilocale
ITensors.op(::OpName"Id:σ+", ::SiteType"vecS=1/2") = I₂ ⊗ σ⁺
ITensors.op(::OpName"σ+:Id", ::SiteType"vecS=1/2") = σ⁺ ⊗ I₂
ITensors.op(::OpName"Id:σ-", ::SiteType"vecS=1/2") = I₂ ⊗ σ⁻
ITensors.op(::OpName"σ-:Id", ::SiteType"vecS=1/2") = σ⁻ ⊗ I₂
# - termini di smorzamento
ITensors.op(::OpName"σx:σx", ::SiteType"vecS=1/2") = σˣ ⊗ σˣ
ITensors.op(::OpName"σx:Id", ::SiteType"vecS=1/2") = σˣ ⊗ I₂
ITensors.op(::OpName"Id:σx", ::SiteType"vecS=1/2") = I₂ ⊗ σˣ

# Composizione dell'Hamiltoniano per i termini di spin
# ----------------------------------------------------
# - termini locali dell'Hamiltoniano
function ITensors.op(::OpName"H1loc", ::SiteType"vecS=1/2", s::Index)
  h = op("σz:Id", s) - op("Id:σz", s)
  return 0.5im * h
end
# - termini bilocali dell'Hamiltoniano
function ITensors.op(::OpName"HspinInt", ::SiteType"vecS=1/2", s1::Index, s2::Index)
  h = op("Id:σ-", s1) * op("Id:σ+", s2) +
      op("Id:σ+", s1) * op("Id:σ-", s2) -
      op("σ-:Id", s1) * op("σ+:Id", s2) -
      op("σ+:Id", s1) * op("σ-:Id", s2)
  return 0.5im * h
end
# - esponenziale del termine che coinvolge solo spin
function ITensors.op(::OpName"expHspin", ::SiteType"vecS=1/2", s1::Index, s2::Index; t::Number, ε::Number)
  ℓ = 0.5ε * op("H1loc", s1) * op("Id:Id", s2) +
      0.5ε * op("Id:Id", s1) * op("H1loc", s2) +
      op("HspinInt", s1, s2)
  return exp(t * ℓ)
end
# - termine di smorzamento per uno spin
function ITensors.op(::OpName"damping", ::SiteType"vecS=1/2", s::Index)
  d = op("σx:σx", s) - op("Id:Id", s)
  return d
end

# Operatori della corrente di spin
# ================================
# Jₖ,ₖ₊₁ = -λ/2 (σˣ₍ₖ₎σʸ₍ₖ₊₁₎ - σʸ₍ₖ₎σˣ₍ₖ₊₁₎)
function J⁺tag(::SiteType"S=1/2", left_site::Int, i::Int)
  # Questa funzione restituisce i nomi degli operatori da assegnare al
  # sito i-esimo per la parte σˣ⊗σʸ di Jₖ,ₖ₊₁ (k ≡ left_site)
  if i == left_site
    str = "σx"
  elseif i == left_site + 1
    str = "σy"
  else
    str = "Id"
  end
  return str
end
function J⁺tag(::SiteType"vecS=1/2", left_site::Int, i::Int)
  if i == left_site
    str = "vecσx"
  elseif i == left_site + 1
    str = "vecσy"
  else
    str = "vecId"
  end
  return str
end

function J⁻tag(::SiteType"S=1/2", left_site::Int, i::Int)
  # Come `J⁺tag`, ma per σʸ⊗σˣ
  if i == left_site
    str = "σy"
  elseif i == left_site + 1
    str = "σx"
  else
    str = "Id"
  end
  return str
end
function J⁻tag(::SiteType"vecS=1/2", left_site::Int, i::Int)
  if i == left_site
    str = "vecσy"
  elseif i == left_site + 1
    str = "vecσx"
  else
    str = "vecId"
  end
  return str
end

function spin_current_op_list(sites::Vector{Index{Int64}})
  N = length(sites)
  # Controllo che i siti forniti siano degli spin ½.
  if all(x -> SiteType("S=1/2") in x, sitetypes.(sites))
    st = SiteType("S=1/2")
    MPtype = MPO
  elseif all(x -> SiteType("vecS=1/2") in x, sitetypes.(sites))
    st = SiteType("vecS=1/2")
    MPtype = MPS
  else
    throw(ArgumentError("spin_current_op_list è disponibile per siti di tipo "*
                        "\"S=1/2\" oppure \"vecS=1/2\"."))
  end
  #
  J⁺ = [MPtype(sites, [J⁺tag(st, k, i) for i = 1:N]) for k = 1:N-1]
  J⁻ = [MPtype(sites, [J⁻tag(st, k, i) for i = 1:N]) for k = 1:N-1]
  return -0.5 .* (J⁺ .- J⁻)
end


# Base di autostati per la catena
# -------------------------------
#=
Come misurare il "contributo" di ogni autospazio dell'operatore numero
N su uno stato rappresentato dalla matrice densità ρ?
Se Pₙ è la proiezione sull'autospazio dell'autovalore n di N, il
valore cercato è cₙ = tr(ρ Pₙ), oppure cₙ = (ψ,Pₙψ⟩ a seconda di com'è
descritto lo stato del sistema.
Il proiettore Pₙ lo posso costruire sommando i proiettori su ogni singolo
autostato nell'n-esimo autospazio: è un metodo grezzo ma dovrebbe
funzionare. Calcolando prima dell'avvio dell'evoluzione temporale tutti
i proiettori si dovrebbe risparmiare tempo, dovento effettuare poi solo
n_sites operazioni ad ogni istante di tempo.                         

La seguente funzione restituisce i nomi per costruire i MPS degli
stati della base dell'intero spazio della catena (suddivisi per
numero di occupazione complessivo).
=#
function chain_basis_states(n_sites::Int, level::Int)
  return unique(permutations([repeat(["Up"], level);
                              repeat(["Dn"], n_sites - level)]))
end

# La seguente funzione crea un proiettore su ciascun sottospazio con
# livello di occupazione definito.
# Metodo "crudo": prendo tutti i vettori della base ortonormale di
# ciascun autospazio, ne creo i proiettori ortogonali e li sommo tutti.
# Forse poco efficiente, ma funziona.
function level_subspace_proj(sites::Vector{Index{Int64}}, level::Int)
  N = length(sites)
  # Controllo che i siti forniti siano degli spin ½.
  if all(x -> SiteType("S=1/2") in x, sitetypes.(sites))
    projs = [projector(MPS(sites, names); normalize=false)
             for names in chain_basis_states(N, level)]
  elseif all(x -> SiteType("vecS=1/2") in x, sitetypes.(sites))
    projs = [MPS(sites, names)
             for names in chain_basis_states(N, level)]
  else
    throw(ArgumentError("spin_current_op_list è disponibile per siti di tipo "*
                        "\"S=1/2\" oppure \"vecS=1/2\"."))
  end
  #return sum(projs) non funziona…
  P = projs[1]
  for p in projs[2:end]
    P += p
  end
  return P
end


# Autostati del primo livello
# ---------------------------
# Potrebbe essere utile anche avere a disposizione la forma degli autostati della
# catena di spin (isolata).
# Gli stati sₖ che hanno il sito k nello stato eccitato e gli altri nello stato
# fondamentale infatti non sono autostati; nella base {sₖ}ₖ (k=1:N) posso
# scrivere l'Hamiltoniano della catena isolata, ristretto all'autospazio di
# singola eccitazione, come
# ε λ 0 0 … 0
# λ ε λ 0 … 0
# 0 λ ε λ … 0
# 0 0 λ ε … 0
# … … … … … …
# 0 0 0 0 … ε
# che ha come autostati vⱼ= ∑ₖ sin(kjπ /(N+1)) sₖ, con j=1:N.
# Attenzione poi a normalizzarli: ‖vⱼ‖² = (N+1)/2.
function single_ex_state(sites::Vector{Index{Int64}}, k::Int)
  N = length(sites)
  if k ∈ 1:N
    states = [i == k ? "Up" : "Dn" for i ∈ 1:N]
  else
    throw(DomainError(k,
                      "Si è tentato di costruire uno stato con eccitazione "*
                      "localizzata nel sito $k, che non appartiene alla "*
                      "catena: inserire un valore tra 1 e $N."))
  end
  return MPS(sites, states)
end

function chain_L1_state(sites::Vector{Index{Int64}}, j::Int)
  # Occhio ai coefficienti: questo, come sopra, non è il vettore vⱼ ma è
  # vec(vₖ⊗vₖᵀ) = vₖ⊗vₖ: di conseguenza i coefficienti della combinazione
  # lineare qui sopra devono essere usati al quadrato.
  # Il prodotto interno tra matrici vettorizzate è tale che
  # ⟨vec(a⊗aᵀ),vec(b⊗bᵀ)⟩ = tr((a⊗aᵀ)ᵀ b⊗bᵀ) = (aᵀb)².
  # Non mi aspetto che la norma di questo MPS sia 1, pur avendo
  # usato i coefficienti normalizzati. Quello che deve fare 1 è invece la
  # somma dei prodotti interni di vec(sₖ⊗sₖ), per k=1:N, con questo vettore.
  # Ottengo poi che ⟨Pₙ,vⱼ⟩ = 1 solo se n=j, 0 altrimenti.
  N = length(sites)
  if j ∉ 1:N
    throw(DomainError(j,
                      "Si è tentato di costruire un autostato della catena "*
                      "con indice $j, che non è valido: bisogna fornire un "*
                      "indice tra 1 e $N."))
  end
  states = [2/(N+1) * sin(j*k*π / (N+1))^2 * single_ex_state(sites, k)
            for k ∈ 1:N]
  return sum(states)
end

# Scelta dello stato iniziale della catena
# ----------------------------------------
# Con un'apposita stringa nei parametri è possibile scegliere lo stato da cui
# far partire la catena di spin. La seguente funzione traduce la stringa
# nell'MPS desiderato, in modo case-insensitive. Le opzioni sono:
# · "empty": stato vuoto
# · "1locM": stato con una (sola) eccitazione localizzata nel sito M ∈ {1,…,N}
# · "1eigM": autostato del primo livello con M ∈ {0,…,N-1} nodi
function parse_init_state(sites::Vector{Index{Int64}}, state::String)
  state = lowercase(state)
  if state == "empty"
    v = MPS(sites, "Dn")
  elseif occursin(r"^1loc", state)
    j = parse(Int, replace(state, "1loc" => ""))
    v = single_ex_state(sites, j)
  elseif occursin(r"^1eig", state)
    j = parse(Int, replace(state, "1eig" => ""))
    # Il j-esimo autostato ha j-1 nodi 
    v = chain_L1_state(sites, j + 1)
  else
    throw(DomainError(state,
                      "Stato non riconosciuto; scegliere tra «empty», «1locN» "*
                      "oppure «1eigN»."))
  end
  return v
end
