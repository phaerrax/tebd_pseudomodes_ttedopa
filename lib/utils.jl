using JSON
using Base.Filesystem
using ITensors
using LinearAlgebra

# Vectorisation utilities
# =======================
# In order to use tr(x'*y) as a tool to extract coefficient the basis must
# of course be orthonormal wrt this inner product.
# The canonical basis or the Gell-Mann one are okay.

"""
    vec(A::Matrix, basis::Vector)

Compute the vector of coefficients of the matrix `A` wrt the basis `basis`.
"""
function vec(A::Matrix, basis::Vector)
  return [tr(b' * A) for b ∈ basis]
end
"""
    vec(L::Function, basis::Vector)

Compute the matrix of coefficients of the linear map `L` wrt the basis `basis`.
The linearity of the map is not checked, so using this function with non-linear
functions leads to undefined results.
"""
function vec(L::Function, basis::Vector)
  return [tr(bi' * L(bj)) for (bi, bj) ∈ Base.product(basis, basis)]
end

"""
    partialtrace(sites::Vector{Index{Int64}}, v::MPS, j::Int)

Compute the partial trace, on the `j`-th site of `sites`, of the matrix
represented, as vectorised, by the MPS `v`.
The result is a `Vector` containing the coordinates of the partial trace
(i.e. the result is still in vectorised form).
"""
function partialtrace(sites::Vector{Index{Int64}}, v::MPS, j::Int)
  # Custom contraction sequence: we start from v[end] and we contract it with
  # a vecId state; we contract the result with v[end-1] and so on, until we
  # get to v[j].  # Then we do the same starting from v[1].
  x = ITensor(1.)
  for i ∈ length(sites):-1:j+1
    x = v[i] * state(sites[i], "vecId") * x
  end
  y = ITensor(1.)
  for i ∈ 1:j-1
    y = v[i] * state(sites[i], "vecId") * y
  end
  z = y * v[j] * x
  # Now `vector(z)` is the coordinate vector of the partial trace.
  return vector(z)
end

# Generalised Gell-Mann matrices
# ==============================
"""
    gellmannmatrix(j, k, dim)

Return the `(j,k)` generalised Gell-Mann matrix of dimension `dim`, normalised
wrt the Hilbert-Schmidt inner product ``(A,B) = tr(A†B)``.
The matrices are indexed as follows:

    * if ``j > k`` the matrix is symmetric and traceless;
    * if ``j < k`` the matrix is antisymmetric;
    * if ``j = k`` the matrix is diagonal.

In particular, ``j = k = dim`` gives a matrix proportional to the identity.
The two indices `j` and `k` determine the non-zero coefficients of the matrix.
The whole set of (different) Gell-Mann matrices that can be generated with this
function is a basis of ``Mat(ℂᵈⁱᵐ)``.
"""
function gellmannmatrix(j, k, dim)
  if j > dim || k > dim || j < 0 || k < 0
    throw(DomainError)
  end
  m = zeros(ComplexF64, dim, dim)
  if j > k
    m[j, k] = 1 / sqrt(2)
    m[k, j] = 1 / sqrt(2)
  elseif k > j
    m[j, k] = -im / sqrt(2)
    m[k, j] = im / sqrt(2)
  elseif j == k && j < dim
    for i ∈ 1:j
      m[i, i] = 1
    end
    m[j+1, j+1] = -j
    m .*= sqrt(1 / (j * (j+1)))
  else
    for i ∈ 1:dim
      m[i, i] = 1/sqrt(dim)
    end
  end
  return m
end

"""
    gellmannbasis(dim)

Return a list containing the "Hermitian basis" of ``Mat(ℂᵈⁱᵐ)``, i.e. composed
of the ``dim²`` generalised Gell-Mann matrices.
"""
function gellmannbasis(dim)
  return [gellmannmatrix(j, k, dim)
          for (j, k) ∈ [Base.product(1:dim, 1:dim)...]]
  # We need to splat the result from `product` so that the result is a list
  # of matrices (a Vector) and not a Matrix.
end

"""
    canonicalmatrix(i, j, dim)

Return the (`i`,`j`) element of the canonical basis of ``Mat(ℂᵈⁱᵐ)``, i.e. a
`dim`×`dim` matrix whose element on the `i`-th row and `j`-th column is ``1``,
and zero elsewhere.
"""
function canonicalmatrix(i, j, dim)
  m = zeros(ComplexF64, dim, dim)
  m[i,j] = 1
  return m
end

"""
    canonicalbasis(dim)

Return a list of the matrices in the canonical basis of ``Mat(ℂᵈⁱᵐ)``. 
The list is ordered corresponding to column-based vectorisation, i.e.

    canonicalbasis(dim)[j] = canonicalmatrix((j-1)%dim + 1, (j-1)÷dim + 1, dim)

with ``j ∈ {1,…,dim²}``. With this ordering,
``vec(A)ⱼ = tr(canonicalbasis(dim)[j]' * A)``.
"""
function canonicalbasis(dim)
  return [canonicalmatrix(i, j, dim)
          for (i, j) ∈ [Base.product(1:dim, 1:dim)...]]
end

# (Von Neumann) Entropy
# =====================
"""
    vonneumannentropy(ψ::MPS, sites::Vector{Index{Int64}}, n::Int)

Compute the entanglement entropy of the biparition ``(1,…,n)|(n+1,…,N)`` of
the system in state described by the MPS `ψ` (defined on the sites `sites`),
using its Schmidt decomposition.
"""
function vonneumannentropy(ψ::MPS, sites::Vector{Index{Int64}}, n::Int)
  orthogonalize!(ψ, n)
  # Decompose ψ[n] in singular values, treating the Link between sites n-1 and n
  # and the physical index as "row index"; the remaining index, the Link
  # between n and n+1, is the "column index".
  _, S, _ = svd(ψ[n], (linkind(ψ, n-1), sites[n]))
  # Compute the square of the singular values (aka the Schmidt coefficients
  # of the bipartition), and from them the entropy.
  sqdiagS = [S[j,j]^2 for j ∈ dim(S, 1)]
  return -sum(p -> p * log(p), sqdiagS; init=0.0)
end

# Chop (from Mathematica)
# =======================
# Imitation of Mathematica's "Chop" function

"""
    chop(x::Real; tolerance=1e-10)

Truncates `x` to zero if it is less than `tolerance`.
"""
function chop(x::Real; tolerance=1e-10)
  return abs(x) > tolerance ? x : zero(x)
end

"""
    chop(x::Complex; tolerance=1e-10)

Truncates the real and/or the imaginary part of `x` to zero if they are less
than `tolerance`.
"""
function chop(x::Complex; tolerance=1e-10)
  return Complex(chop(real(x)), chop(imag(x)))
end

# Manipulating ITensor objects
# ============================

"""
    sitetypes(s::Index)

Return the ITensor tags of `s` as SiteTypes. 

This function is already defined in the ITensor library, but it is not publicly
accessible.
"""
function sitetypes(s::Index)
  ts = tags(s)
  return SiteType[SiteType(ts.data[n]) for n in 1:length(ts)]
end

# Reading the parameters of the simulations
# =========================================

"""
    isjson(filename::String)

Check if `filename` ends in ".json".

By design, filenames consisting of only ".json" return `false`.
"""
function isjson(filename::String)
  return length(filename) > 5 && filename[end-4:end] == ".json"
  # We ignore a file whose name is only ".json".
end

"""
    load_parameters(file_list)

Load the JSON files contained in `file_list` into dictionaries, returning a
list of dictionaries, one for each file.

If `file_list` is a filename, then the list will contain just one dictionary;
if `file_list` is a directory, every JSON file within it is loaded and a
dictionary is created for each one of them.
"""
function load_parameters(file_list)
  if isempty(file_list)
    throw(ErrorException("No parameter file provided."))
  end
  first_arg = file_list[1]
  prev_dir = pwd()
  if isdir(first_arg)
    # If the first argument is a directory, we read all the JSON files within
    # and load them as parameter files; in the end all output will be saved
    # in that same directory. We ignore the remaining elements in ARGS.
    cd(first_arg)
    files = filter(isjson, readdir())
    @info "$first_arg is a directory. Ignoring other command line arguments."
  else
    # Otherwise, all command line arguments are treated as parameter files.
    # Output files will be saved in the pwd() from which the script was
    # launched.
    files = file_list
  end
  # Load parameters into dictionaries, one for each file.
  parameter_lists = []
  for f in files
    open(f) do input
      s = read(input, String)
      # Add the filename too to the dictionary.
      push!(parameter_lists, merge(Dict("filename" => f), JSON.parse(s)))
    end
  end
  cd(prev_dir)
  return parameter_lists
end

function allequal(a)
  return all(x -> x == first(a), a)
end

# Defining the time interval of the simulation
# ============================================

"""
    construct_step_list(parameters)

Return a list of time instants at which the time evolution will be evaluated.

The values run from zero up to ``parameters["simulation_end_time"]``, with a
step size equal to ``parameters["simulation_time_step"]``.
"""
function construct_step_list(parameters)
  τ = parameters["simulation_time_step"]
  end_time = parameters["simulation_end_time"]
  return collect(range(0, end_time; step=τ))
end

# Computing expectation values of observables
# ===========================================

# Function that calculates the eigenvalues of the number operator, given a set
# of projectors on the eigenspaces, and also return their sum (which should
# be 1).
# TODO: these two functions are probably outdated. Anyway they are not
# specific to the occupation levels, so they should have a more general
# description.
function levels(projs::Vector{MPO}, state::MPS)
  lev = [real(inner(state, p * state)) for p in projs]
  return [lev; sum(lev)]
end
function levels(projs::Vector{MPS}, state::MPS)
  lev = [real(inner(p, state)) for p in projs]
  return [lev; sum(lev)]
end

# MPS and MPO utilities
# =====================

"""
    linkdims(m::Union{MPS, MPO})

Return a list of the bond dimensions of `m`.
"""
function linkdims(m::Union{MPS, MPO})
  return [ITensors.dim(linkind(m, j)) for j ∈ 1:length(m)-1]
end

"""
    chain(left::MPS, right::MPS)

Concatenate `left` and `right`, returning `left` ``⊗`` `right`.
"""
function chain(left::MPS, right::MPS)
  # This function is like mpnum's `chain`: it takes two MPSs ans concatenates
  # them. The code is "inspired" from `ITensors.MPS` in mps.jl:308.

  midN = length(left) # The site with the missing link between the two MPSs.
  # First of all we shift the Link tags of the given MPSs, so that the final
  # enumeration of the tags is correct.
  # Note that in each MPS the numbers in the Link tags do not follow the
  # numbering of the Sites on which it is based, they always start from 1.
  for j in eachindex(left)
    replacetags!(left[j], "l=$j", "l=$j"; tags="Link")
    replacetags!(left[j], "l=$(j-1)", "l=$(j-1)"; tags="Link")
  end
  for j in eachindex(right)
    replacetags!(right[j], "l=$j", "l=$(midN+j)"; tags="Link")
    replacetags!(right[j], "l=$(j-1)", "l=$(midN+j-1)"; tags="Link")
  end
  # "Shallow" concatenation of the MPSs (there's still a missing link).
  M = MPS([left..., right...])
  # We create a "trivial" Index of dimension 1 and add it to the two sites
  # which are not yet connected.
  # The Index has dimension 1 because this is a tensor product between the
  # states so there's no correlation between them.
  missing_link = Index(1; tags="Link,l=$midN")
  M[midN] = M[midN] * state(missing_link, 1)
  M[midN+1] = state(dag(missing_link), 1) * M[midN+1]

  return M
end

"""
    chain(left::MPO, right::MPO)

Concatenate `left` and `right`, returning `left` ``⊗`` `right`.
"""
function chain(left::MPO, right::MPO)
  # Like the previous `chain`, but for MPOs.
  midN = length(left)# The site with the missing link between the two MPOs.
  for j in eachindex(right)
    replacetags!(right[j], "l=$j", "l=$(midN+j)"; tags="Link")
    replacetags!(right[j], "l=$(j-1)", "l=$(midN+j-1)"; tags="Link")
  end
  M = MPO([left..., right...])
  missing_link = Index(1; tags="Link,l=$midN")
  M[midN] = M[midN] * state(missing_link, 1)
  M[midN+1] = M[midN+1] * state(dag(missing_link), 1)
  # The order of the Indexes in M[midN] and M[midN+1] ns not what we would get
  # if we built an MPO on the whole system as usual; namely, the two "Link"
  # Indexes are swapped.
  # This however should not matter since ITensor does not care about the order.
  return M
end

# Varargs versions

"""
    chain(a::MPS, b...)

Concatenate the given MPSs into a longer MPS, returning their tensor product.
"""
chain(a::MPS, b...) = chain(a, chain(b...))
"""
    chain(a::MPO, b...)

Concatenate the given MPOs into a longer MPO, returning their tensor product.
"""
chain(a::MPO, b...) = chain(a, chain(b...))

"""
    embed_slice(sites::Array{Index{Int64}}, range::UnitRange{Int}, slice::MPO)

Embed `slice`, defined on a subset `range` of `sites`, into a MPO which covers
the whole `sites`.

The MPO is extended by filling the empty spots with an "Id" operator, therefore
an operator with OpName "Id" is required to be defined for the SiteTypes of the
remaining sites.

# Arguments
- `sites::Array{Index{Int64}}`: the sites of the whole system.
- `range::UnitRange{Int}`: the range spanned by `slice`.
- `slice::MPO`: the MPO to be extended.
"""
function embed_slice(sites::Array{Index{Int64}},
                     range::UnitRange{Int},
                     slice::MPO)
  # TODO: compute automatically on which sites the MPO is defined, without
  # having to supply the range explicitly as an argument.
  if length(slice) != length(range)
    throw(DimensionMismatch("slice and range must have the same size."))
  end
  if !issubset(range, eachindex(sites))
    throw(BoundsError(range, sites))
  end

  if range[begin] == 1 && range[end] == length(sites)
    mpo = slice
  elseif range[begin] == 1
    mpo = chain(slice,
                MPO(sites[range[end]+1 : end], "Id"))
  elseif range[end] == length(sites)
    mpo = chain(MPO(sites[1 : range[begin]-1], "Id"),
                slice)
  else
    mpo = chain(MPO(sites[1 : range[begin]-1], "Id"),
                slice,
                MPO(sites[range[end]+1 : end], "Id"))
  end
  return mpo
end

"""
    embed_slice(sites::Array{Index{Int64}}, range::UnitRange{Int}, slice::MPS)

Embed `slice`, defined on a subset `range` of `sites`, into a MPS which covers
the whole `sites` (to be interpreted as a vectorised operator).

The MPS is extended by filling the empty spots with a "vecId" operator,
therefore an operator with OpName "vecId" is required to be defined for the
SiteTypes of the remaining sites.

# Arguments
- `sites::Array{Index{Int64}}`: the sites of the whole system.
- `range::UnitRange{Int}`: the range spanned by `slice`.
- `slice::MPS`: the MPS to be extended.
"""
function embed_slice(sites::Array{Index{Int64}},
                     range::UnitRange{Int},
                     slice::MPS)
  if length(slice) != length(range)
    throw(DimensionMismatch("slice e range must have the same size."))
  end
  if !issubset(range, eachindex(sites))
    throw(BoundsError(range, sites))
  end

  if range[begin] == 1 && range[end] == length(sites)
    mpo = slice
  elseif range[begin] == 1
    mpo = chain(slice,
                MPS(sites[range[end]+1 : end], "vecId"))
  elseif range[end] == length(sites)
    mpo = chain(MPS(sites[1 : range[begin]-1], "vecId"),
                slice)
  else
    mpo = chain(MPS(sites[1 : range[begin]-1], "vecId"),
                slice,
                MPS(sites[range[end]+1 : end], "vecId"))
  end
  return mpo
end
