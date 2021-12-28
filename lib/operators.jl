using ITensors

# Costruzione della lista di operatori 2-locali
function twositeoperators(sites::Vector{Index{Int64}},
                          localcfs::Vector{<:Real},
                          interactioncfs::Vector{<:Real})
  # Restituisce la lista di termini (che dovranno essere poi esponenziati nel
  # modo consono alla simulazione) dell'Hamiltoniano o del “Lindbladiano” che
  # agiscono sulle coppie di siti adiacenti della catena.
  # L'elemento list[j] è l'operatore hⱼ,ⱼ₊₁ (o ℓⱼ,ⱼ₊₁, a seconda della notazione)
  #
  # Argomenti
  # ---------
  # · `sites::Vector{Index}`: un vettore di N elementi, contenente gli Index
  #   che rappresentano i siti del sistema;
  # · `localcfs::Vector{Index}`: un vettore di N elementi contenente i
  #   coefficienti che moltiplicano i termini locali dell'Hamiltoniano o del
  #   Lindbladiano
  # · `interactioncfs::Vector{Index}`: un vettore di N-1 elementi contenente i
  #   coefficienti che moltiplicano i termini di interazione tra siti adiacenti,
  #   con la convenzione che l'elemento j è riferito al termine hⱼ,ⱼ₊₁/ℓⱼ,ⱼ₊₁.
  list = ITensor[]
  localcfs[begin] *= 2
  localcfs[end] *= 2
  # Anziché dividere per casi il ciclo che segue distinguendo i siti ai lati
  # della catena (che non devono avere il fattore 0.5 scritto sotto), moltiplico
  # qui per 2 i rispettivi coefficienti degli operatori locali.
  for j ∈ 1:length(sites)-1
    s1 = sites[j]
    s2 = sites[j+1]
    h = 0.5localcfs[j] * localop(s1) * op("Id", s2) +
        0.5localcfs[j+1] * op("Id", s1) * localop(s2) +
        interactioncfs[j] * interactionop(s1, s2)
    push!(list, h)
  end
  return list
end

# Operatori locali
function localop(s::Index)
  if SiteType("S=1/2") ∈ sitetypes(s)
    t = 0.5op("σz", s)
  elseif SiteType("Osc") ∈ sitetypes(s)
    t = op("N", s)
  elseif SiteType("vecS=1/2") ∈ sitetypes(s)
    t = 0.5im * (op("σz:Id", s) - op("Id:σz", s))
  elseif SiteType("HvS=1/2") ∈ sitetypes(s)
    t = -0.5im * (op("σz⋅", s) - op("⋅σz", s))
  elseif SiteType("vecOsc") ∈ sitetypes(s)
    t = im * (op("N:Id", s) - op("Id:N", s))
  elseif SiteType("HvOsc") ∈ sitetypes(s)
    t = -im * (op("N⋅", s) - op("⋅N", s))
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return t
end

# Operatori di interazione
function interactionop(s1::Index, s2::Index)
  if SiteType("S=1/2") ∈ sitetypes(s1) && SiteType("S=1/2") ∈ sitetypes(s2)
    t = -0.5 * (op("σ-", s1) * op("σ+", s2) +
                op("σ+", s1) * op("σ-", s2))
  elseif SiteType("Osc") ∈ sitetypes(s1) && SiteType("S=1/2") ∈ sitetypes(s2)
    t = op("X", s1) * op("σx", s2)
  elseif SiteType("S=1/2") ∈ sitetypes(s1) && SiteType("Osc") ∈ sitetypes(s2)
    t = op("σx", s1) * op("X", s2)
  elseif SiteType("Osc") ∈ sitetypes(s1) && SiteType("Osc") ∈ sitetypes(s2)
    t = (op("a+", s1) * op("a-", s2) +
         op("a-", s1) * op("a+", s2))
  #
  elseif SiteType("vecS=1/2") ∈ sitetypes(s1) &&
         SiteType("vecS=1/2") ∈ sitetypes(s2)
    t = -0.5im * (op("σ-:Id", s1) * op("σ+:Id", s2) +
                  op("σ+:Id", s1) * op("σ-:Id", s2) -
                  op("Id:σ-", s1) * op("Id:σ+", s2) -
                  op("Id:σ+", s1) * op("Id:σ-", s2))
  elseif SiteType("vecOsc") ∈ sitetypes(s1) &&
         SiteType("vecS=1/2") ∈ sitetypes(s2)
    t = im * (op("asum:Id", s1) * op("σx:Id", s2) -
              op("Id:asum", s1) * op("Id:σx", s2))
  elseif SiteType("vecS=1/2") ∈ sitetypes(s1) &&
         SiteType("vecOsc") ∈ sitetypes(s2)
    t = im * (op("σx:Id", s1) * op("asum:Id", s2) -
              op("Id:σx", s1) * op("Id:asum", s2))
  #
  elseif SiteType("HvS=1/2") ∈ sitetypes(s1) &&
         SiteType("HvS=1/2") ∈ sitetypes(s2)
    t = 0.5im * (op("σ-⋅", s1) * op("σ+⋅", s2) +
                 op("σ+⋅", s1) * op("σ-⋅", s2) -
                 op("⋅σ-", s1) * op("⋅σ+", s2) -
                 op("⋅σ+", s1) * op("⋅σ-", s2))
  elseif SiteType("HvOsc") ∈ sitetypes(s1) &&
         SiteType("HvS=1/2") ∈ sitetypes(s2)
    t = -im * (op("asum⋅", s1) * op("σx⋅", s2) -
               op("⋅asum", s1) * op("⋅σx", s2))
  elseif SiteType("HvS=1/2") ∈ sitetypes(s1) &&
         SiteType("HvOsc") ∈ sitetypes(s2)
    t = -im * (op("σx⋅", s1) * op("asum⋅", s2) -
               op("⋅σx", s1) * op("⋅asum", s2))
  else
    throw(DomainError((s1, s2), "SiteType non riconosciuti."))
  end
  return t
end
