using LinearAlgebra
using ITensors
using Combinatorics

include("utils.jl")

const ⊗ = kron

# Pauli matrices and the like
# ---------------------------
σˣ = [0 1; 1 0]
σʸ = [0 -im; im 0]
σᶻ = [1 0; 0 -1]
σ⁺ = [0 1; 0 0]
σ⁻ = [0 0; 1 0]
I₂ = [1 0; 0 1]
ê₊ = [1; 0]
ê₋ = [0; 1]

# Additional ITensor states and operators (in addition to the ones already
# defined for S=1/2 sites):
# - null vector
ITensors.state(::StateName"0", ::SiteType"S=1/2") = [0; 0]
# - number operator, aka |↑⟩⟨↑|
ITensors.op(::OpName"N", st::SiteType"S=1/2") = op(OpName"ProjUp"(), st)
# - identity matrices
ITensors.op(::OpName"Id", ::SiteType"S=1/2") = I₂
# - null matrix
ITensors.op(::OpName"0", ::SiteType"S=1/2") = zeros(2, 2)
# - ladder operators (aliases)
ITensors.op(::OpName"σ+", st::SiteType"S=1/2") = op(OpName("S+"), st)
ITensors.op(::OpName"σ-", st::SiteType"S=1/2") = op(OpName("S-"), st)
ITensors.op(::OpName"plus", st::SiteType"S=1/2") = op(OpName("S+"), st)
ITensors.op(::OpName"minus", st::SiteType"S=1/2") = op(OpName("S-"), st)

# Space of spin-1/2 particles (vectorised)
# ========================================

"""
    ITensors.space(::SiteType"vecS=1/2")

Create the Hilbert space for a site of type "vecS=1/2", i.e. a vectorised
spin-1/2 particle, where the vectorisation is performed wrt the canonical
basis of `Mat(ℂ²)`.
"""
ITensors.space(::SiteType"vecS=1/2") = 4

# States
# ------

ITensors.state(::StateName"Up", ::SiteType"vecS=1/2") = ê₊ ⊗ ê₊
ITensors.state(::StateName"Dn", ::SiteType"vecS=1/2") = ê₋ ⊗ ê₋

# States representing vectorised operators
# ----------------------------------------

function ITensors.state(::StateName"vecσx", ::SiteType"vecS=1/2")
  return vec(σˣ, canonicalbasis(2))
end
function ITensors.state(::StateName"vecσy", ::SiteType"vecS=1/2")
  return vec(σʸ, canonicalbasis(2))
end
function ITensors.state(::StateName"vecσz", ::SiteType"vecS=1/2")
  return vec(σᶻ, canonicalbasis(2))
end
function ITensors.state(::StateName"vecId", ::SiteType"vecS=1/2")
  return vec(I₂, canonicalbasis(2))
end
function ITensors.state(::StateName"vecN", ::SiteType"vecS=1/2")
  return vec([1 0; 0 0], canonicalbasis(2))
end
function ITensors.state(::StateName"vec0", ::SiteType"vecS=1/2")
  return vec([0 0; 0 0], canonicalbasis(2))
end

function ITensors.state(::StateName"vecX", st::SiteType"vecS=1/2")
  return ITensors.state(StateName("vecσx"), st)
end
function ITensors.state(::StateName"vecY", st::SiteType"vecS=1/2")
  return ITensors.state(StateName("vecσy"), st)
end
function ITensors.state(::StateName"vecZ", st::SiteType"vecS=1/2")
  return ITensors.state(StateName("vecσz"), st)
end

function ITensors.state(::StateName"vecplus", ::SiteType"vecS=1/2")
  return vec(σ⁺, canonicalbasis(2))
end
function ITensors.state(::StateName"vecminus", ::SiteType"vecS=1/2")
  return vec(σ⁻, canonicalbasis(2))
end

# Operators acting on vectorised oscillators
# ------------------------------------------

# For vectorisation wrt the canonical basis, there exist explicit formula for
# the representazione of linear maps, namely
#    ρ ↦ A ρ B
# is transformed into the map
#    vec(ρ) ↦ (B ⊗ Aᵀ) vec(ρ)
# on the respective vectorised state.

# - identiy
ITensors.op(::OpName"Id:Id", ::SiteType"vecS=1/2") = I₂ ⊗ I₂
ITensors.op(::OpName"Id", ::SiteType"vecS=1/2") = I₂ ⊗ I₂
# - local Hamiltonian terms
ITensors.op(::OpName"σz:Id", ::SiteType"vecS=1/2") = σᶻ ⊗ I₂
ITensors.op(::OpName"Id:σz", ::SiteType"vecS=1/2") = I₂ ⊗ σᶻ
# - interaction Hamiltonian terms
ITensors.op(::OpName"Id:σ+", ::SiteType"vecS=1/2") = I₂ ⊗ σ⁺
ITensors.op(::OpName"σ+:Id", ::SiteType"vecS=1/2") = σ⁺ ⊗ I₂
ITensors.op(::OpName"Id:σ-", ::SiteType"vecS=1/2") = I₂ ⊗ σ⁻
ITensors.op(::OpName"σ-:Id", ::SiteType"vecS=1/2") = σ⁻ ⊗ I₂
# - damping terms in GKSL equation
#   1) symmetric damping
ITensors.op(::OpName"σx:σx", ::SiteType"vecS=1/2") = σˣ ⊗ σˣ
ITensors.op(::OpName"σx:Id", ::SiteType"vecS=1/2") = σˣ ⊗ I₂
ITensors.op(::OpName"Id:σx", ::SiteType"vecS=1/2") = I₂ ⊗ σˣ
ITensors.op(::OpName"Damping", ::SiteType"vecS=1/2") = (σˣ ⊗ σˣ) - (I₂ ⊗ I₂)
#   2) asymmetric damping with separate absorption/dissipation terms
function ITensors.op(::OpName"Damping2", ::SiteType"vecS=1/2"; ω::Number, T::Number)
  if T == 0
    n = 0
  else
    n = (ℯ^(ω / T) - 1)^(-1)
  end
  d = vec(x -> (n + 1) * (σ⁻*x*σ⁺ - 0.5*(σ⁺*σ⁻*x + x*σ⁺*σ⁻)) +
               n * (σ⁺*x*σ⁻ - 0.5*(σ⁻*σ⁺*x + x*σ⁻*σ⁺)),
          canonicalbasis(2))
  return d
end

# Space of spin-1/2 particles (vectorised wrt Gell-Mann matrices)
# ===============================================================

"""
    ITensors.space(st::SiteType"HvS=1/2"; dim = 2)

Create the Hilbert space for a site of type "HvS=1/2", i.e. a vectorised
spin-1/2 particle, where the vectorisation is performed wrt the generalised
Gell-Mann basis of `Mat(ℂ²)`, composed of Hermitian traceless matrices
together with the identity matrix.
"""
ITensors.space(::SiteType"HvS=1/2") = 4

# Elements of and operators on Mat(ℂ²) are expanded wrt the basis {Λᵢ}ᵢ₌₁⁴ of
# generalised Gell-Mann matrices (plus a multiple of the identity).
# An element A ∈ Mat(ℂ²) is representeb by the a vector v such that
#     vᵢ = tr(Λᵢ A),
# while a linear map L : Mat(ℂ²) → Mat(ℂ²) by the matrix ℓ such that
#     ℓᵢⱼ = tr(Λᵢ L(Λⱼ)).

# States
# ------

# "Up" ≡ ê₊ ⊗ ê₊'
# "Dn" ≡ ê₋ ⊗ ê₋'
function ITensors.state(::StateName"Up", ::SiteType"HvS=1/2")
  return vec(ê₊ ⊗ ê₊', gellmannbasis(2))
end
function ITensors.state(::StateName"Dn", ::SiteType"HvS=1/2")
  return vec(ê₋ ⊗ ê₋', gellmannbasis(2))
end

# States representing vectorised operators
# ----------------------------------------

function ITensors.state(::StateName"vecσx", ::SiteType"HvS=1/2")
  return vec(σˣ, gellmannbasis(2))
end
function ITensors.state(::StateName"vecσy", ::SiteType"HvS=1/2")
  return vec(σʸ, gellmannbasis(2))
end
function ITensors.state(::StateName"vecσz", ::SiteType"HvS=1/2")
  return vec(σᶻ, gellmannbasis(2))
end

function ITensors.state(::StateName"vecId", ::SiteType"HvS=1/2")
  return vec(I₂, gellmannbasis(2))
end
function ITensors.state(::StateName"vecN", ::SiteType"HvS=1/2")
  return vec([1 0; 0 0], gellmannbasis(2))
end
function ITensors.state(::StateName"vec0", ::SiteType"HvS=1/2")
  return vec(zeros(2, 2), gellmannbasis(2))
end

function ITensors.state(::StateName"vecX", st::SiteType"HvS=1/2")
  return ITensors.state(StateName("vecσx"), st)
end
function ITensors.state(::StateName"vecY", st::SiteType"HvS=1/2")
  return ITensors.state(StateName("vecσy"), st)
end
function ITensors.state(::StateName"vecZ", st::SiteType"HvS=1/2")
  return ITensors.state(StateName("vecσz"), st)
end

function ITensors.state(::StateName"vecplus", ::SiteType"HvS=1/2")
  return vec(σ⁺, gellmannbasis(2))
end
function ITensors.state(::StateName"vecminus", ::SiteType"HvS=1/2")
  return vec(σ⁻, gellmannbasis(2))
end

# Operators acting on vectorised spins
# ------------------------------------

# Luckily, even when they are acting on two sites at the same times, every
# operator we need to define is factorised (or a sum of factorised operators).
# This simplifies immensely the calculations: if
#   L : Mat(ℂ²) ⊗ Mat(ℂ²) → Mat(ℂ²) ⊗ Mat(ℂ²)
# can be written as L₁ ⊗ L₂ for Lᵢ : Mat(ℂ²) → Mat(ℂ²) then
#   ⟨êᵢ₁ ⊗ êᵢ₂, L(êⱼ₁ ⊗ êⱼ₂)⟩ = ⟨êᵢ₁, L₁(êⱼ₁)⟩ ⟨êᵢ₂, L₂(êⱼ₂)⟩.

function ITensors.op(::OpName"Id", ::SiteType"HvS=1/2")
  return vec(identity, gellmannbasis(2))
end

function ITensors.op(s::OpName"σ+⋅", ::SiteType"HvS=1/2")
  return vec(x -> σ⁺*x, gellmannbasis(2))
end
function ITensors.op(::OpName"⋅σ+", ::SiteType"HvS=1/2")
  return vec(x -> x*σ⁺, gellmannbasis(2))
end

function ITensors.op(s::OpName"σ-⋅", ::SiteType"HvS=1/2")
  return vec(x -> σ⁻*x, gellmannbasis(2))
end
function ITensors.op(::OpName"⋅σ-", ::SiteType"HvS=1/2")
  return vec(x -> x*σ⁻, gellmannbasis(2))
end

function ITensors.op(s::OpName"σx⋅", ::SiteType"HvS=1/2")
  return vec(x -> σˣ*x, gellmannbasis(2))
end
function ITensors.op(::OpName"⋅σx", ::SiteType"HvS=1/2")
  return vec(x -> x*σˣ, gellmannbasis(2))
end

function ITensors.op(s::OpName"σz⋅", ::SiteType"HvS=1/2")
  return vec(x -> σᶻ*x, gellmannbasis(2))
end
function ITensors.op(::OpName"⋅σz", ::SiteType"HvS=1/2")
  return vec(x -> x*σᶻ, gellmannbasis(2))
end

# Spin current operators 
# ======================

function J⁺tag(::SiteType"S=1/2", left_site::Int, i::Int)
  # TODO: still useful?
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
function J⁺tag(::SiteType"HvS=1/2", left_site::Int, i::Int)
  return J⁺tag(SiteType("vecS=1/2"), left_site, i)
end

function J⁻tag(::SiteType"S=1/2", left_site::Int, i::Int)
  # Just as `J⁺tag`, but for σʸ⊗σˣ
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
function J⁻tag(::SiteType"HvS=1/2", left_site::Int, i::Int)
  return J⁻tag(SiteType("vecS=1/2"), left_site, i)
end

function spin_current_op_list(sites::Vector{Index{Int64}})
  N = length(sites)
  # Check if all sites are spin-½ sites.
  if all(x -> SiteType("S=1/2") in x, sitetypes.(sites))
    st = SiteType("S=1/2")
    MPtype = MPO
  elseif all(x -> SiteType("vecS=1/2") in x, sitetypes.(sites))
    st = SiteType("vecS=1/2")
    MPtype = MPS
  elseif all(x -> SiteType("HvS=1/2") in x, sitetypes.(sites))
    st = SiteType("HvS=1/2")
    MPtype = MPS
  else
    throw(ArgumentError("spin_current_op_list works with SiteTypes "*
                        "\"S=1/2\", \"vecS=1/2\" or \"HvS=1/2\"."))
  end
  #
  J⁺ = [MPtype(sites, [J⁺tag(st, k, i) for i = 1:N]) for k = 1:N-1]
  J⁻ = [MPtype(sites, [J⁻tag(st, k, i) for i = 1:N]) for k = 1:N-1]
  return -0.5 .* (J⁺ .- J⁻)
end


# Chain eigenstate basis
# ----------------------

# How to measure how much each eigenspace of the number operator (of the
# whole spin chain) "contributes" to a given state ρ?
# We build a projector operator associated to each eigenspace.
# The projector on the m-th level eigenspace can be made by simply adding the
# orthogonal projections on each element of the canonical basis which has m
# "Up" spins and N-m "Down" spins, N being the length of the chain.
# This may result in a very big MPO. If these are needed for more than one
# simulation, calculating them once and for all before the simulations start
# may help to cut down the computation time.

"""
    chain_basis_states(n::Int, level::Int)

Return a list of strings which can be used to build the MPS of all states in
the ``ℂ²ⁿ`` basis that contain `level` "Up" spins.
"""
function chain_basis_states(n::Int, level::Int)
  return unique(permutations([repeat(["Up"], level);
                              repeat(["Dn"], n - level)]))
end

"""
    level_subspace_proj(sites::Vector{Index{Int64}}, l::Int)

Return the projector on the subspace with `l` "Up" spins.
"""
function level_subspace_proj(sites::Vector{Index{Int64}}, l::Int)
  N = length(sites)
  # Check if all sites are spin-½ sites.
  if all(x -> SiteType("S=1/2") in x, sitetypes.(sites))
    projs = [projector(MPS(sites, names); normalize=false)
             for names in chain_basis_states(N, l)]
  elseif all(x -> SiteType("vecS=1/2") in x, sitetypes.(sites)) ||
         all(x -> SiteType("HvS=1/2") in x, sitetypes.(sites))
    projs = [MPS(sites, names)
             for names in chain_basis_states(N, l)]
  else
    throw(ArgumentError("level_subspace_proj works with SiteTypes "*
                        "\"S=1/2\", \"vecS=1/2\" or \"HvS=1/2\"."))
  end
  # Somehow return sum(projs) doesn't work… we have to sum manually.
  P = projs[1]
  for p in projs[2:end]
    P = +(P, p; cutoff=1e-10)
  end
  return P
end


# First-level chain eigenstates
# -----------------------------

# It is useful to have at hand the eigenstates of the free chain Hamiltonian
# within the first-level eigenspace of the number operator ([H,N] = 0).
# States |sₖ⟩ with an up spin on site k and a down spin on the others are
# in fact not eigenstates, but they still form a basis for the eigenspace;
# wrt the {sₖ}ₖ (k ∈ {1,…,N}) basis the free chain Hamiltonian is written as
#   ⎛ ε λ 0 0 … 0 ⎞
#   ⎜ λ ε λ 0 … 0 ⎟
#   ⎜ 0 λ ε λ … 0 ⎟
#   ⎜ 0 0 λ ε … 0 ⎟
#   ⎜ ⋮ ⋮ ⋮ ⋮ ⋱ ⋮ ⎟
#   ⎝ 0 0 0 0 … ε ⎠
# whose eigenstates are |vⱼ⟩= ∑ₖ sin(kjπ /(N+1)) |sₖ⟩, con j ∈ {1,…,N}.
# Note that they are not normalised: ‖|vⱼ⟩‖² = (N+1)/2.

"""
    single_ex_state(sites::Vector{Index{Int64}}, k::Int)

Return the MPS of a state with a single excitation on the `k`-th site.
"""
function single_ex_state(sites::Vector{Index{Int64}}, k::Int)
  N = length(sites)
  if k ∈ 1:N
    states = [i == k ? "Up" : "Dn" for i ∈ 1:N]
  else
    throw(DomainError(k,
                      "Trying to build a state with an excitation localised "*
                      "at site $k, which does not belong to the chain: please "*
                      "insert a value between 1 and $N."))
  end
  return MPS(sites, states)
end

"""
    chain_L1_state(sites::Vector{Index{Int64}}, j::Int)

Return the `j`-th eigenstate of the chain Hamiltonian in the single-excitation
subspace.
"""
function chain_L1_state(sites::Vector{Index{Int64}}, j::Int)
  # FIXME: this seems just wrong. Why not computing directly vec(|vⱼ⟩ ⊗ ⟨vⱼ|), 
  # without going through the linear combination?
  #
  # Careful with the coefficients: this isn't |vⱼ⟩ but vec(|vⱼ⟩ ⊗ ⟨vⱼ|),
  # so if we want to build it as a linear combination of the |sₖ⟩'s we
  # need to square the coefficients.
  # Note that vectorised projectors satisfy
  #     ⟨vec(a⊗aᵀ), vec(b⊗bᵀ)⟩ = tr((a⊗aᵀ)ᵀ b⊗bᵀ) = (aᵀb)².
  # We don't expect the norm of this MPS to be 1. What has to be 1, is the
  # sum of the inner products of vec(|sₖ⟩ ⊗ ⟨sₖ|) and this vector, for
  # k ∈ {1,…,N}. We get ⟨Pₙ|vⱼ⟩ = 1 only if n=j, otherwise it is 0.
  N = length(sites)
  if j ∉ 1:N
    throw(DomainError(j,
                      "Trying to build a chain eigenstate with invalid index "*
                      "$j: please insert a value between 1 and $N."))
  end
  states = [2/(N+1) * sin(j*k*π / (N+1))^2 * single_ex_state(sites, k)
            for k ∈ 1:N]
  return sum(states)
end

# Choice of the spin chain's initial state
# ----------------------------------------

"""
    parse_init_state(sites::Vector{Index{Int64}}, state::String)

Return an MPS representing a particular state of the spin chain, given
by the string `state`:

  * "empty" -> empty state (aka the ground state of the chain Hamiltonian)
  * "1locM" -> state with a single excitation at site `M` (``M ∈ {1,…,N}``)
  * "1eigM" -> single-excitation eigenstate of the chain Hamiltonian with `M` nodes (``M ∈ {0,…,N-1}``)

The string is case-insensitive. The length `N` of the chain is computed
from `sites`. 
"""
function parse_init_state(sites::Vector{Index{Int64}}, state::String)
  state = lowercase(state)
  if state == "empty"
    v = MPS(sites, "Dn")
  elseif occursin(r"^1loc", state)
    j = parse(Int, replace(state, "1loc" => ""))
    v = single_ex_state(sites, j)
  elseif occursin(r"^1eig", state)
    j = parse(Int, replace(state, "1eig" => ""))
    # The j-th eigenstate has j-1 nodes
    v = chain_L1_state(sites, j + 1)
  else
    throw(DomainError(state,
                      "Unrecognised state: please choose from \"empty\", "*
                      "\"1locN\" or \"1eigN\"."))
  end
  return v
end

"""
    parse_spin_state(site::Index{Int64}, state::String)

Return the MPS of a single spin site representing a particular state, given
by the string `state`:

- "empty", "dn", "down" → spin-down state
- "up"                  → spin-up state
- "x+"                  → ``1/√2 ( |+⟩ + |-⟩ )`` state
"""
function parse_spin_state(site::Index{Int64}, state::String)
  state = lowercase(state)
  if state == "empty" || state == "dn" || state == "down"
    v = ITensors.state(site, "Dn")
  elseif state == "up"
    v = ITensors.state(site, "Up")
  elseif state == "x+"
    v = 1/sqrt(2) * (ITensors.state(site, "Up") + ITensors.state(site, "Dn"))
  else
    throw(DomainError(state,
                      "Unrecognised state: please choose from \"empty\",
                      \"up\", \"down\" or \"x+\"."))
  end
  return MPS([v])
end
