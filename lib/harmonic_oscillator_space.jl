using ITensors
using LinearAlgebra

const ⊗ = kron

# In this file the states and operators describing harmonic oscillators (in a
# T-TEDOPA setting) are defined, both in a normal and vectorised way.
# We base the normal version on the "Qudit" SiteType provided by ITensors.

# Basic matrices
# --------------

"""
    a⁻(dim::Int)

Return the matrix representing the destruction operator of a harmonic oscillator
of the given dimension `dim`, in the eigenbasis of the number operator.
"""
a⁻(dim::Int) = diagm(1 => [sqrt(j) for j = 1:dim-1])

"""
    a⁻(dim::Int)

Return the matrix representing the creation operator of a harmonic oscillator
of the given dimension `dim`, in the eigenbasis of the number operator. The
truncation to a finite dimension is performed by imposing ``a⁺|dim-1⟩ = 0`` on
the most-occupied state in the
basis.
"""
a⁺(dim::Int) = diagm(-1 => [sqrt(j) for j = 1:dim-1])

"""
    num(dim::Int)

Return the matrix representing the number operator of a harmonic oscillator of
the given dimension `dim`, in its eigenbasis.
"""
num(dim::Int) = a⁺(dim) * a⁻(dim)

"""
    id(dim::Int)

Return the identity operator of a harmonic oscillator of the given dimension
`dim`.
"""
id(dim::Int) = Matrix{Int}(I, dim, dim)

"""
	oscdimensions(N, basedim, decay)

Compute a decreasing sequence of length `N` where the first two elements are
equal to `basedim` and the following ones are given by
`floor(2 + basedim * ℯ^(-decay * n))`.

Useful to determine the dimensions of oscillator sites in a TEDOPA chain.
"""
function oscdimensions(length, basedim, decay)
  f(j) = 2 + basedim * ℯ^(-decay * j)
  return [basedim; basedim; (Int ∘ floor ∘ f).(3:length)]
end

# Space of harmonic oscillators
# =============================

alias(::SiteType"Osc") = SiteType"Qudit"()

"""
    ITensors.space(st::SiteType"Osc";
                   dim = 2,
                   conserve_qns = false,
                   conserve_number = false,
                   qnname_number = "Number")

Create the Hilbert space for a site of type "Osc".

Optionally specify the conserved symmetries and their quantum number labels.
"""
ITensors.space(st::SiteType"Osc"; kwargs...) = space(alias(st); kwargs...)

# States
# ------

# Generic function that forwards "Osc" state definitions to "Qudit" states.
# The available states are therefore the same between the two SiteTypes.
ITensors.state(sn::StateName, st::SiteType"Osc", s::Index) = state(sn, alias(st), s)

# Operators
# ---------

# When we call
#   op(name::AbstractString, s::Index...; kwargs...)
# the following functions are called by the library (in this order):
#   op(::OpName, ::SiteType, ::Index; kwargs...)
#   op(::OpName, ::SiteType; kwargs...)
# By defining the following function, we place ourselves before the first of
# these two calls, "intercepting" the process so that we can add the dimension
# of the state to the keyword arguments; then we call `op` again and resume
# the normal execution, with the new kwargs.
# We also need to wrap the result in an `itensor` object so that the operator
# is really an ITensor (see for example src/physics/site_types/qudit.jl:73)
# since the custom `op` function we define below just return matrices.
function ITensors.op(on::OpName, st::SiteType"Osc", s::Index; kwargs...)
  return itensor(op(on, st; dim=ITensors.dim(s), kwargs...), s', dag(s))
end

ITensors.op(::OpName"a+", ::SiteType"Osc"; dim=2) = a⁺(dim)
ITensors.op(::OpName"a-", ::SiteType"Osc"; dim=2) = a⁻(dim)
ITensors.op(::OpName"plus", st::SiteType"Osc"; kwargs...) = ITensors.op(OpName("a+"), st; kwargs...)
ITensors.op(::OpName"minus", st::SiteType"Osc"; kwargs...) = ITensors.op(OpName("a-"), st; kwargs...)

ITensors.op(::OpName"Id", ::SiteType"Osc"; dim=2) = id(dim)
ITensors.op(::OpName"N", ::SiteType"Osc"; dim=2) = num(dim)
ITensors.op(::OpName"X", ::SiteType"Osc"; dim=2) = a⁻(dim) + a⁺(dim)
ITensors.op(::OpName"Y", ::SiteType"Osc"; dim=2) = im*(a⁻(dim) - a⁺(dim))

# Space of harmonic oscillators (vectorised)
# ==========================================

"""
    ITensors.space(st::SiteType"vecOsc"; dim = 2)

Create the Hilbert space for a site of type "vecOsc", i.e. a vectorised
harmonic oscillator of dimension `dim` (this means that the space has dimension
`dim^2`), where the vectorisation is performed wrt the canonical basis of
`Mat(ℂᵈⁱᵐ)`.
"""
function ITensors.space(::SiteType"vecOsc"; dim=2)
  return dim^2
end

# States
# ------

# Here there is no "vecQudit" on which we can depend for the states, so we
# need to make our own from scratch.
# Each state is ultimately a vector whose length is `dim^2`, and we can get
# `dim` by calling `ITensors.dim` on the state's Index.
# We do as we did for the `op` function above for the "Osc" SiteType, i.e.
#   state(s::Index, name::AbstractString; kwargs...)
# tries to call, in turn,
#    state(::StateName"Name", ::SiteType"Tag", s::Index; kwargs...)
#    state(::StateName"Name", ::SiteType"Tag"; kwargs...)
# so we try to intercept the call to the first one, we add the dimension to
# the keyword arguments, and finally we call the second one, which will refer
# to one of the specific `op`s we define below.
function ITensors.state(sn::StateName, st::SiteType"vecOsc", s::Index; kwargs...)
  return state(sn, st; dim=isqrt(ITensors.dim(s)), kwargs...)
  # Use `isqrt` which takes an Int and returns an Int; the decimal part is
  # truncated, but it's not a problem since by construction dim(s) is a
  # perfect square, so the decimal part should be zero.
end

# There's a problem with this trick: before calling `state` as explained above,
# ITensors also tries to call
#   state(::StateName"Name", st::SiteType"Tag", s::Index; kwargs...)
# for each Tag `st` in the Index `s`. If `nothing` is returned, then it goes on
# to the next available Tag, exhausting all possibilities; if the return value
# is not `nothing` then the result is passed to `itensor` and encapsulated into
# an ITensor object.
# The point is, if we create "vecOsc" sites through `siteinds` they will have
# the Tags "Site" and "n=N" (for some `N`) too, and there is no
#   state(::StateName{ThermEq}, ::SiteType{Site}, ::Index{Int64}; kwargs...)
# (for example) or with "n=N". As a consequence, a MethodError exception is
# thrown.
# To solve this, we define a sort of "default" function which always returns
# `nothing`, on which `state` can fall back if there is no other `state`
# defined.
ITensors.state(sn::StateName, st::SiteType, s::Index; kwargs...) = nothing

function ITensors.state(::StateName{N}, ::SiteType"vecOsc"; dim=2) where {N}
  # Eigenstates of the number operator (NOT the canonical basis of ℂᵈⁱᵐ ⊗ ℂᵈⁱᵐ)
  n = parse(Int, String(N))
  v = zeros(dim)
  v[n + 1] = 1.0
  return v ⊗ v
end

function ITensors.state(::StateName"ThermEq", st::SiteType"vecOsc"; dim=2, ω::Real, T::Real)
  # Thermal equilibrium state at given frequency and temperature.
  # TODO: is there a way we can include ω and T directy within the "vecOsc"
  # Index? After all, they are set at the beginning, and never change.
  if T == 0
    v = state(StateName("0"), st; dim=dim)
  else
    mat = exp(-ω / T * num(dim))
    mat /= tr(mat)
    v = vcat(mat[:])
  end
  return v
end

# FIXME: fix the signature of the function (j and k should be mandatory args).
# Or maybe just delete the function. Do we really need this?
function ITensors.state(::StateName"mat_comp", ::SiteType"vecOsc"; dim=2, j::Int, k::Int)
  êⱼ = zeros(dim)
  êₖ = zeros(dim)
  êⱼ[j + 1] = 1.0
  êₖ[k + 1] = 1.0
  return êⱼ ⊗ êₖ
end

# States representing vectorised operators
# ----------------------------------------

# Expectation values of observables are translated into the inner product of
# the vectorised matrices, i.e.
#   ⟨A⟩ = tr(A ρ) = vec(A)† vec(ρ)
# so we need to define vec(A) for some known observable As.

function ITensors.state(::StateName"veca+", ::SiteType"vecOsc"; dim=2)
  # This is not an observable by itself, but it may be used to build proper
  # observables which span more than one site (i.e. current operators).
  return vec(a⁺(dim), canonicalbasis(dim))
end
function ITensors.state(::StateName"veca-", ::SiteType"vecOsc"; dim=2)
  return vec(a⁻(dim), canonicalbasis(dim))
end
function ITensors.state(::StateName"vecN", ::SiteType"vecOsc"; dim=2)
  return vec(num(dim), canonicalbasis(dim))
end
function ITensors.state(::StateName"vecId", ::SiteType"vecOsc"; dim=2)
  return vec(id(dim), canonicalbasis(dim))
end

function ITensors.state(::StateName"vecplus", st::SiteType"vecOsc"; kwargs...)
  return ITensors.state(StateName("veca+"), st; kwargs...)
end
function ITensors.state(::StateName"vecminus", st::SiteType"vecOsc"; kwargs...)
  return ITensors.state(StateName("veca-"), st; kwargs...)
end

# Operators acting on vectorised oscillators
# ------------------------------------------

# Same as `op` above with "Osc" SiteTypes.
function ITensors.op(on::OpName, st::SiteType"vecOsc", s::Index; kwargs...)
  return itensor(op(on, st; dim=isqrt(ITensors.dim(s)), kwargs...), s', dag(s))
end

# For vectorisation wrt the canonical basis, there exist explicit formula for
# the representazione of linear maps, namely
#    ρ ↦ A ρ B
# is transformed into the map
#    vec(ρ) ↦ (B ⊗ Aᵀ) vec(ρ)
# on the respective vectorised state.

ITensors.op(::OpName"Id:Id", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ id(dim)
ITensors.op(::OpName"Id", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ id(dim)
# - spin-oscillator interaction terms
ITensors.op(::OpName"Id:asum", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ (a⁺(dim)+a⁻(dim))
ITensors.op(::OpName"asum:Id", ::SiteType"vecOsc"; dim=2) = (a⁺(dim)+a⁻(dim)) ⊗ id(dim)
# - free Hamiltonian
ITensors.op(::OpName"N:Id", ::SiteType"vecOsc"; dim=2) = num(dim) ⊗ id(dim)
ITensors.op(::OpName"Id:N", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ num(dim)
# - terms appearing in dissipative operators in GKSL equation
ITensors.op(::OpName"a-a+:Id", ::SiteType"vecOsc"; dim=2) = (a⁻(dim)*a⁺(dim)) ⊗ id(dim)
ITensors.op(::OpName"Id:a-a+", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ (a⁻(dim)*a⁺(dim))
ITensors.op(::OpName"a+T:a-", ::SiteType"vecOsc"; dim=2) = transpose(a⁺(dim)) ⊗ a⁻(dim)
ITensors.op(::OpName"a-T:a+", ::SiteType"vecOsc"; dim=2) = transpose(a⁻(dim)) ⊗ a⁺(dim)

# Space of harmonic oscillators (vectorised wrt Gell-Mann matrices)
# =================================================================

"""
    ITensors.space(st::SiteType"HvOsc"; dim = 2)

Create the Hilbert space for a site of type "HvOsc", i.e. a vectorised
harmonic oscillator of dimension `dim` (this means that the space has dimension
`dim^2`), where the vectorisation is performed wrt the generalised Gell-Mann
basis of `Mat(ℂᵈⁱᵐ)`, composed of Hermitian traceless matrices together with
the identity matrix.
"""
function ITensors.space(::SiteType"HvOsc"; dim=2)
  return dim^2
end
# TODO: find out if there's a way we can unify "vecOsc" and "HvOsc" in a single
# SiteType, passing the choice of basis as a parameter at the moment of the
# creation of the site (as we do with the dimension).

# States
# ------

# The following functions are mostly the same as those for "vecOsc", except
# that we use the Gell-Mann (Hermitian) basis.

function ITensors.state(sn::StateName, st::SiteType"HvOsc", s::Index; kwargs...)
  return state(sn, st; dim=isqrt(ITensors.dim(s)), kwargs...)
end

function ITensors.state(::StateName{N}, ::SiteType"HvOsc"; dim=2) where {N}
  # Eigenstates êₙ ⊗ êₙ of the number operator, written wrt the Hermitian
  # basis.
  n = parse(Int, String(N))
  v = zeros(dim)
  v[n + 1] = 1.0
  return vec(v ⊗ v', gellmannbasis(dim))
end

function ITensors.state(::StateName"ThermEq", st::SiteType"HvOsc"; dim=2, ω, T)
  if T == 0
    v = state(StateName("0"), st; dim=dim)
  else
    mat = exp(-ω / T * num(dim))
    mat /= tr(mat)
    v = vec(mat, gellmannbasis(dim))
  end
  return v
end

# Product of X = a+a† and of the thermal equilibrium state Z⁻¹vec(exp(-βH)).
# It is used in the computation of the correlation function of the bath.
function ITensors.state(::StateName"X⋅Therm", st::SiteType"HvOsc"; dim=2, ω, T)
  if T == 0
    mat = zeros(Float64, dim, dim)
    mat[1, 1] = 1.0
  else
    mat = exp(-ω / T * num(dim))
    mat /= tr(mat)
  end
  return vec((a⁺(dim) + a⁻(dim)) * mat, gellmannbasis(dim))
end

function ITensors.state(::StateName"mat_comp", ::SiteType"HvOsc"; dim=2, j::Int, k::Int)
  êⱼ = zeros(dim)
  êₖ = zeros(dim)
  êⱼ[j + 1] = 1.0
  êₖ[k + 1] = 1.0
  return vec(êⱼ ⊗ êₖ', gellmannbasis(dim))
end

# States representing vectorised operators
# ----------------------------------------

function ITensors.state(::StateName"veca+", ::SiteType"HvOsc"; dim=2)
  return vec(a⁺(dim), gellmannbasis(dim))
end
function ITensors.state(::StateName"veca-", ::SiteType"HvOsc"; dim=2)
  return vec(a⁻(dim), gellmannbasis(dim))
end
function ITensors.state(::StateName"vecN", ::SiteType"HvOsc"; dim=2)
  return vec(num(dim), gellmannbasis(dim))
end
function ITensors.state(::StateName"vecId", ::SiteType"HvOsc"; dim=2)
  return vec(id(dim), gellmannbasis(dim))
end

function ITensors.state(::StateName"vecX", ::SiteType"HvOsc"; dim=2)
  return vec(a⁻(dim) + a⁺(dim), gellmannbasis(dim))
end
function ITensors.state(::StateName"vecY", ::SiteType"HvOsc"; dim=2)
  return vec(im*(a⁻(dim) - a⁺(dim)), gellmannbasis(dim))
end

function ITensors.state(::StateName"vecplus", st::SiteType"HvOsc"; kwargs...)
  return ITensors.state(StateName("veca+"), st; kwargs...)
end
function ITensors.state(::StateName"vecminus", st::SiteType"HvOsc"; kwargs...)
  return ITensors.state(StateName("veca-"), st; kwargs...)
end

# Operators acting on vectorised oscillators
# ------------------------------------------

function ITensors.op(on::OpName, st::SiteType"HvOsc", s::Index; kwargs...)
  return itensor(op(on, st; dim=isqrt(ITensors.dim(s)), kwargs...), s', dag(s))
end

function ITensors.op(::OpName"Id", ::SiteType"HvOsc"; dim=2)
  return vec(identity, gellmannbasis(dim))
end

function ITensors.op(::OpName"⋅a+", ::SiteType"HvOsc"; dim=2)
  return vec(x -> x*a⁺(dim), gellmannbasis(dim))
end
function ITensors.op(::OpName"a+⋅", ::SiteType"HvOsc"; dim=2)
  return vec(x -> a⁺(dim)*x, gellmannbasis(dim))
end

function ITensors.op(::OpName"⋅a-", ::SiteType"HvOsc"; dim=2)
  return vec(x -> x*a⁻(dim), gellmannbasis(dim))
end
function ITensors.op(::OpName"a-⋅", ::SiteType"HvOsc"; dim=2)
  return vec(x -> a⁻(dim)*x, gellmannbasis(dim))
end

function ITensors.op(::OpName"⋅asum", ::SiteType"HvOsc"; dim=2)
  return vec(x -> x*(a⁺(dim)+a⁻(dim)), gellmannbasis(dim))
end
function ITensors.op(::OpName"asum⋅", ::SiteType"HvOsc"; dim=2)
  return vec(x -> (a⁺(dim)+a⁻(dim))*x, gellmannbasis(dim))
end

function ITensors.op(::OpName"N⋅", ::SiteType"HvOsc"; dim=2)
  return vec(x -> num(dim)*x, gellmannbasis(dim))
end
function ITensors.op(::OpName"⋅N", ::SiteType"HvOsc"; dim=2)
  return vec(x -> x*num(dim), gellmannbasis(dim))
end

# GKSL equation terms
# -------------------

# Dissipator in the canonical basis
function ITensors.op(::OpName"Damping", ::SiteType"vecOsc", s::Index; ω::Number, T::Number)
  if T == 0
    n = 0
  else
    n = (ℯ^(ω / T) - 1)^(-1)
  end
  d = (n + 1) * (op("a+T:a-", s) - 0.5 * (op("N:Id", s) + op("Id:N", s))) +
      n * (op("a-T:a+", s) - 0.5 * (op("a-a+:Id", s) + op("Id:a-a+", s)))
  return d
end

# Dissipator in the Hermitian basis
function ITensors.op(::OpName"Damping", ::SiteType"HvOsc"; dim=2, ω::Number, T::Number)
  if T == 0
    n = 0
  else
    n = (ℯ^(ω / T) - 1)^(-1)
  end
  A = a⁻(dim)
  A⁺ = a⁺(dim)
  d = vec(x -> (n + 1) * (A*x*A⁺ - 0.5*(A⁺*A*x + x*A⁺*A)) +
               n * (A⁺*x*A - 0.5*(A*A⁺*x + x*A*A⁺)),
          gellmannbasis(dim))
  return d
end

# Separate absorption and dissipation terms in GKSL equation
function ITensors.op(::OpName"Lindb+", ::SiteType"HvOsc"; dim=2)
  A = a⁻(dim)
  A⁺ = a⁺(dim)
  d = vec(x -> A*x*A⁺ - 0.5*(A⁺*A*x + x*A⁺*A), gellmannbasis(dim))
  return d
end
function ITensors.op(::OpName"Lindb-", ::SiteType"HvOsc"; dim=2)
  A = a⁻(dim)
  A⁺ = a⁺(dim)
  d = vec(x -> A⁺*x*A - 0.5*(A*A⁺*x + x*A*A⁺),
          gellmannbasis(dim))
  return d
end

# Mixed dissipator appearing in the equation for two pseudomodes
function mixedlindbladplus(s1::Index{Int64}, s2::Index{Int64})
  return (op("a-⋅", s1) * op("⋅a+", s2) +
          op("a-⋅", s2) * op("⋅a+", s1) -
          0.5*(op("a+⋅", s1) * op("a-⋅", s2) +
               op("a+⋅", s2) * op("a-⋅", s1)) -
          0.5*(op("⋅a+", s1) * op("⋅a-", s2) +
               op("⋅a+", s2) * op("⋅a-", s1)))
end
function mixedlindbladminus(s1::Index{Int64}, s2::Index{Int64})
  return (op("a+⋅", s1) * op("⋅a-", s2) +
          op("a+⋅", s2) * op("⋅a-", s1) -
          0.5*(op("a-⋅", s1) * op("a+⋅", s2) +
               op("a-⋅", s2) * op("a+⋅", s1)) -
          0.5*(op("⋅a-", s1) * op("⋅a+", s2) +
               op("⋅a-", s2) * op("⋅a+", s1)))
end

# Projection on the eigenstates of the number operator
# ------------------------------------------------

# TODO: this is a really basic function... we could do without this.
"""
    osc_levels_proj(s::Index{Int64}, n::Int)

Return an MPS representing the `n`-th occupied level of `s`'s SiteType.
"""
function osc_levels_proj(site::Index{Int64}, level::Int)
  st = state(site, "$level")
  return MPS([st])
end

# Choice of the oscillator's initial state
# ----------------------------------------

"""
    parse_init_state_osc(site::Index{Int64},
                         statename::String;
                         <keyword arguments>)

Return an MPS representing a particular state of a harmonic oscillator, given
by the string `statename`:

- "thermal" → thermal equilibrium state
- "fockN"   → `N`-th eigenstate of the number operator (element of Fock basis)
- "empty"   → alias for "fock0"

The string is case-insensitive. Other parameters required to build the state
(e.g. frequency, temperature) may be supplied as keyword arguments.
"""
function parse_init_state_osc(site::Index{Int64}, statename::String; kwargs...)
  # TODO: maybe remove "init" from title? It is a generic state, after all.
  statename = lowercase(statename)
  if statename == "thermal"
    s = state(site, "ThermEq"; kwargs...)
  elseif occursin(r"^fock", statename)
    j = parse(Int, replace(statename, "fock" => ""))
    s = state(site, "$j")
  elseif statename == "empty"
    s = state(site, "0")
  else
    throw(DomainError(statename,
                      "Unrecognised state name; please choose from "*
                      "\"empty\", \"fockN\" or \"thermal\"."))
  end
  return MPS([s])
end
