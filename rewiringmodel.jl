using Pkg
Pkg.activate("Rewiring")
# Pkg.instantiate() # to install necessary packages.

using DifferentialEquations
using Graphs
using GraphPlot
using Compose
using Cairo
using Colors
using DataFrames
using CSV
using Statistics
using LinearAlgebra
using Random
using Plots
using StatsBase
using Base.Threads
using JLD2
using Dates
using Distributions
using ProgressMeter

# ============================================================================
# SIMULATION PARAMETERS - CONFIGURATION
# ============================================================================

# Functional response type
const FUNCTIONAL_RESPONSE = :type_II  # Options: :type_I, :type_II, :type_III

# Rewiring toggles
ENABLE_EXTINCTION_REWIRING = false  # Rewire when last prey goes extinct
ENABLE_ENERGY_REWIRING = false      # Rewire when energy intake < mortality
ENABLE_DYNAMIC_REPLACEMENT = false   # Randomly replace least profitable link (Kondoh Eq. 2)
ENABLE_INVASION = false             # Causes solver issues - currently disabled

# Species capabilities
FRACTION_CAN_REWIRE = 1.0           # Fraction of consumers that can rewire (0.0-1.0)
FRACTION_CAN_ADAPT = 1.0         # Fraction of consumers that can adapt foraging (0.0-1.0)

# Rewiring parameters
const ENERGY_THRESHOLD_FACTOR = 1.0      # θ: rewire when intake < θ * mortality 
const ENERGY_REWIRING_INTERVAL = 10.0    # How frequently to check energy rewiring condition.
const DYNAMIC_REPLACEMENT_RATE = 0.1     # Events per time unit per consumer (increased from 0.01)
const REWIRING_COST = 0.25               # Reduction in attack rate for new prey (0-1)

# Rewiring constraints (to prevent unrealistic rapid rewiring)
const REWIRING_COOLDOWN = 10.0           # Minimum time between rewiring events per species
const MAX_REWIRING_PER_SPECIES = 5       # Maximum rewiring events per species (lifetime limit)

# Diet breadth parameters (controls when energy rewiring ADDS vs REPLACES)
const SPECIALIST_MAX_PREY = 1            # Consumer with ≤ this many prey is a "specialist" - default 1
const GENERALIST_MAX_PREY = 3            # Consumer stops adding prey at this limit - default 3
# Logic: 
#   n_prey ≤ SPECIALIST_MAX_PREY → Energy rewiring ADDS prey (expand niche)
#   SPECIALIST_MAX_PREY < n_prey < GENERALIST_MAX_PREY → Energy rewiring REPLACES prey (optimize)
#   n_prey ≥ GENERALIST_MAX_PREY → Energy rewiring REPLACES prey (at limit)

# Rewiring constraints (to prevent unrealistic rewiring)
const USE_TROPHIC_CONSTRAINTS = true     # Only rewire to species at lower trophic level
const USE_BODY_MASS_CONSTRAINTS = true   # Only rewire to prey within mass ratio range
const MIN_PREDATOR_PREY_MASS_RATIO = 1.0 # Predator must be >= this times prey mass
const MAX_PREDATOR_PREY_MASS_RATIO = 100.0 # Predator must be <= this times prey mass

# Adaptation parameters
ADAPTATION_RATE = 0.25                   # G: speed of foraging effort adjustment

# Invasion parameters
const INVASION_RATE = 0.01               # Probability per time unit
const INVASION_CHECK_INTERVAL = 10.0     # How often to check for invasions

# Environmental stochasticity
const STOCHASTICITY_ENABLED = true
const STOCHASTICITY_INTERVAL = 10.0      # Time between perturbations
const STOCHASTICITY_INTENSITY = 0.3      # Magnitude of perturbations

# Extinction threshold
const EXTINCTION_THRESHOLD = 1e-6

# Simulation parameters
USE_RANDOM_INITIAL = true             # If true, skip burn-in
const BURNIN_TIME = 200.0
const SIMULATION_TIME = 5000.0
const SWEEP_SIMULATION_TIME = 250.0      # For parameter sweeps
const SAVE_INTERVAL = 1.0

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@enum RewiringType begin
    EXTINCTION_FORCED
    ENERGY_DEFICIT
    DYNAMIC_REPLACEMENT
end

mutable struct FoodWebParameters
    S::Int                          # Number of species
    n_basal::Int                    # Number of basal species
    r::Vector{Float64}              # Growth rates (basal species)
    d::Vector{Float64}              # Mortality rates (consumers)
    K::Vector{Float64}              # Carrying capacities (basal species)
    attack_rates::Matrix{Float64}   # Attack rates f_ij
    conversion_eff::Matrix{Float64} # Conversion efficiencies e_ij
    handling_times::Matrix{Float64} # Handling times T_ij
    q::Matrix{Float64}              # Capture exponents (for Type III)
    adjacency::Matrix{Bool}         # Who eats whom
    is_basal::Vector{Bool}          # Is species a basal resource?
    can_rewire::Vector{Bool}        # Can this consumer rewire?
    can_adapt::Vector{Bool}         # Can this consumer adapt foraging efforts?
    is_extinct::Vector{Bool}        # Track extinction status
    initial_species::Set{Int}       # Species present at t=0
    trophic_level::Vector{Float64}  # Trophic level (1=basal, 2=herbivore, etc) - Float64 for fractional
    body_mass::Vector{Float64}      # Body mass (scales with trophic level)
    last_rewiring_time::Vector{Float64}  # Time of last rewiring for each species
    rewiring_count::Vector{Int}     # Number of times each species has rewired
end

mutable struct AdaptiveState
    foraging_efforts::Matrix{Float64}  # a_ij: foraging effort allocation
    original_attack_rates::Matrix{Float64}  # Store original rates for rewiring cost
end

struct RewiringEvent
    time::Float64
    event_type::RewiringType
    consumer_id::Int
    old_prey_id::Union{Int,Nothing}
    new_prey_id::Int
    similarity_consumer::Union{Int,Nothing}
    is_random::Bool
    profitability::Union{Float64,Nothing}
end

# ============================================================================
# FOOD WEB GENERATION
# ============================================================================

function generate_cascade_foodweb(S::Int, C::Float64; n_basal::Int=div(S, 3))
    """
    Generate a cascade food web with S species and connectance C.
    """
    adjacency = zeros(Bool, S, S)

    for i in (n_basal+1):S
        for j in 1:(i-1)
            if rand() < C
                adjacency[i, j] = true
            end
        end
    end

    # Ensure each consumer has at least one prey
    for i in (n_basal+1):S
        if sum(adjacency[i, :]) == 0
            j = rand(1:(i-1))
            adjacency[i, j] = true
        end
    end

    return adjacency
end

function generate_random_foodweb(S::Int, C::Float64; n_basal::Int=div(S, 3))
    """
    Generate a random food web with S species and connectance C.
    """
    adjacency = zeros(Bool, S, S)

    for i in (n_basal+1):S
        for j in 1:S
            if i != j && rand() < C
                adjacency[i, j] = true
            end
        end
    end

    # Ensure each consumer has at least one prey
    for i in (n_basal+1):S
        if sum(adjacency[i, :]) == 0
            j = rand(1:S)
            while j == i
                j = rand(1:S)
            end
            adjacency[i, j] = true
        end
    end

    return adjacency
end

function initialize_parameters(adjacency::Matrix{Bool};
    model_type::Symbol=:cascade)
    """
    Initialize food web parameters based on adjacency matrix.
    """
    S = size(adjacency, 1)
    n_basal = findfirst(i -> sum(adjacency[i, :]) > 0, 1:S) - 1
    if isnothing(n_basal)
        n_basal = S
    end

    is_basal = [i <= n_basal for i in 1:S]

    # Growth rates for basal species
    r = zeros(S)
    r[1:n_basal] .= rand(Uniform(0.005, 0.05), n_basal)
    # r[1:n_basal] .= rand(Uniform(0.01, 0.1), n_basal) #Gilljam default r ~ U(0.01,0.1)

    # Mortality rates for consumers 
    d = zeros(S)
    d[(n_basal+1):S] .= rand(Uniform(0.001, 0.005), S - n_basal)
    # d[(n_basal+1):S] .= rand(Uniform(0.001, 0.01), S - n_basal) #Gilljam default d ~ U(0.001,0.01)

    # Carrying capacities for basal species 
    K = zeros(S)
    K[1:n_basal] .= rand(Uniform(1.0, 3.0), n_basal)

    # Attack rates 
    attack_rates = zeros(S, S)
    for i in 1:S, j in 1:S
        if adjacency[i, j]
            attack_rates[i, j] = rand(Uniform(0.8, 2.0)) #default (0.8,2.0)
        end
    end

    # Conversion efficiency
    conversion_eff = zeros(S, S)
    for i in 1:S, j in 1:S
        if adjacency[i, j]
            conversion_eff[i, j] = rand(Uniform(0.2, 0.4))
        end
    end

    # Handling times
    handling_times = zeros(S, S)
    if FUNCTIONAL_RESPONSE != :type_I
        for i in 1:S, j in 1:S
            if adjacency[i, j]
                handling_times[i, j] = rand(Uniform(0.2, 0.6))
            end
        end
    end

    # Capture exponents (for Type III functional response)
    q = ones(S, S)
    if FUNCTIONAL_RESPONSE == :type_III
        for i in 1:S, j in 1:S
            if adjacency[i, j]
                q[i, j] = rand(Uniform(1.0, 2.0))
            end
        end
    end

    # Randomly assign rewiring and adaptation capabilities
    can_rewire = [!is_basal[i] && rand() < FRACTION_CAN_REWIRE for i in 1:S]
    can_adapt = [!is_basal[i] && rand() < FRACTION_CAN_ADAPT for i in 1:S]

    # Track extinction status
    is_extinct = falses(S)
    initial_species = Set(1:S)

    # Calculate trophic levels 
    trophic_level = [get_trophic_level(i, adjacency) for i in 1:S]

    # Assign body masses based on trophic level with some variation
    # Base mass increases with √10 per level (3.16x per level)
    # Plus random variation within each level (0.5x to 2x)
    body_mass = zeros(S)
    for i in 1:S
        base_mass = 10.0^((trophic_level[i] - 1.0) * 0.5)  # Gentler scaling: √10 per level
        variation = rand(Uniform(0.5, 2.0))
        body_mass[i] = base_mass * variation
    end

    # Initialize rewiring tracking
    last_rewiring_time = fill(-Inf, S)  # -Inf means never rewired
    rewiring_count = zeros(Int, S)      # Count of rewiring events

    return FoodWebParameters(S, n_basal, r, d, K, attack_rates,
        conversion_eff, handling_times, q, adjacency, is_basal,
        can_rewire, can_adapt, is_extinct, initial_species,
        trophic_level, body_mass, last_rewiring_time, rewiring_count)
end

function initialize_foraging_efforts(params::FoodWebParameters)
    """
    Initialize foraging efforts equally among prey.
    """
    efforts = zeros(params.S, params.S)

    for i in 1:params.S
        prey_indices = findall(params.adjacency[i, :])
        if length(prey_indices) > 0
            efforts[i, prey_indices] .= 1.0 / length(prey_indices)
        end
    end

    return AdaptiveState(efforts, copy(params.attack_rates))
end

# ============================================================================
# DYNAMICS - ROSENZWEIG-MACARTHUR WITH ADAPTIVE FORAGING
# ============================================================================

function functional_response(N_j::Float64, f_ij::Float64, T_ij::Float64,
    q_ij::Float64, consumer_i::Int,
    all_prey_densities::Vector{Float64},
    foraging_efforts::Vector{Float64},
    attack_rates::Vector{Float64},
    handling_times::Vector{Float64})
    """
    Compute functional response based on type.
    """
    if FUNCTIONAL_RESPONSE == :type_I
        return f_ij * N_j
    elseif FUNCTIONAL_RESPONSE == :type_II
        denominator = 1.0
        for k in 1:length(all_prey_densities)
            if foraging_efforts[k] > 0 && all_prey_densities[k] > EXTINCTION_THRESHOLD
                denominator += handling_times[k] * foraging_efforts[k] *
                               attack_rates[k] * all_prey_densities[k]
            end
        end
        return f_ij * N_j / denominator
    else  # :type_III
        denominator = 1.0
        for k in 1:length(all_prey_densities)
            if foraging_efforts[k] > 0 && all_prey_densities[k] > EXTINCTION_THRESHOLD
                denominator += handling_times[k] * foraging_efforts[k] *
                               attack_rates[k] * (all_prey_densities[k]^q_ij)
            end
        end
        return f_ij * (N_j^q_ij) / denominator
    end
end

function foodweb_dynamics!(du, u, p, t)
    """
    Rosenzweig-MacArthur dynamics with adaptive foraging (Kondoh 2003).
    u = [N_1, ..., N_S, a_11, a_12, ..., a_SS]
    """
    params, state = p
    S = params.S

    # Extract populations and foraging efforts
    N = @view u[1:S]
    a_flat = @view u[(S+1):end]
    a = reshape(a_flat, S, S)

    # Population dynamics
    for i in 1:S
        du[i] = 0.0

        # Skip extinct species
        if N[i] < EXTINCTION_THRESHOLD
            du[i] = 0.0
            continue
        end

        if params.is_basal[i]
            # Basal species: logistic growth
            du[i] += params.r[i] * N[i] * (1 - N[i] / params.K[i])
        else
            # Consumer: mortality
            du[i] -= params.d[i] * N[i]
        end

        # Loss due to predation
        for j in 1:S
            if params.adjacency[j, i] && N[i] > EXTINCTION_THRESHOLD && N[j] > EXTINCTION_THRESHOLD
                prey_densities = [N[k] for k in 1:S]
                foraging_efforts_j = [a[j, k] for k in 1:S]
                attack_rates_j = [params.attack_rates[j, k] for k in 1:S]
                handling_times_j = [params.handling_times[j, k] for k in 1:S]

                FR = functional_response(N[i], params.attack_rates[j, i],
                    params.handling_times[j, i],
                    params.q[j, i], j,
                    prey_densities, foraging_efforts_j,
                    attack_rates_j, handling_times_j)

                du[i] -= a[j, i] * FR * N[j]
            end
        end

        # Gain from consumption
        if !params.is_basal[i]
            for j in 1:S
                if params.adjacency[i, j] && N[j] > EXTINCTION_THRESHOLD && N[i] > EXTINCTION_THRESHOLD
                    prey_densities = [N[k] for k in 1:S]
                    foraging_efforts_i = [a[i, k] for k in 1:S]
                    attack_rates_i = [params.attack_rates[i, k] for k in 1:S]
                    handling_times_i = [params.handling_times[i, k] for k in 1:S]

                    FR = functional_response(N[j], params.attack_rates[i, j],
                        params.handling_times[i, j],
                        params.q[i, j], i,
                        prey_densities, foraging_efforts_i,
                        attack_rates_i, handling_times_i)

                    du[i] += params.conversion_eff[i, j] * a[i, j] * FR * N[j]
                end
            end
        end

        # # Prevent negative densities
        # if N[i] + du[i] * 0.01 < 0  # Check with small dt
        #     du[i] = -N[i] / 0.01  # Ensure we don't go negative
        # end
    end

    # Adaptive foraging dynamics (Kondoh Eq. 2) - only if ADAPTATION_RATE > 0
    idx = S + 1
    for i in 1:S
        if !params.is_basal[i] && params.can_adapt[i] && N[i] > EXTINCTION_THRESHOLD && ADAPTATION_RATE > 0
            prey_indices = findall(params.adjacency[i, :])

            if length(prey_indices) > 0
                # Calculate average profitability
                avg_profit = 0.0
                for j in prey_indices
                    if N[j] > EXTINCTION_THRESHOLD
                        profit_j = params.conversion_eff[i, j] *
                                   params.attack_rates[i, j] * N[j]
                        avg_profit += a[i, j] * profit_j
                    end
                end

                # Update foraging efforts
                for j in 1:S
                    if params.adjacency[i, j]
                        if N[j] > EXTINCTION_THRESHOLD
                            profit_j = params.conversion_eff[i, j] *
                                       params.attack_rates[i, j] * N[j]
                            du[idx] = ADAPTATION_RATE * a[i, j] * (profit_j - avg_profit)
                        else
                            du[idx] = 0.0
                        end
                    else
                        du[idx] = 0.0
                    end
                    idx += 1
                end
            else
                # No prey, freeze all efforts
                for j in 1:S
                    du[idx] = 0.0
                    idx += 1
                end
            end
        else
            # Basal species or cannot adapt - freeze foraging efforts
            for j in 1:S
                du[idx] = 0.0
                idx += 1
            end
        end
    end
end

# ============================================================================
# REWIRING FUNCTIONS
# ============================================================================

function calculate_jaccard_similarity(consumer1_prey::Vector{Int},
    consumer2_prey::Vector{Int})
    """
    Calculate Jaccard similarity between two consumers' prey sets.
    """
    if length(consumer1_prey) == 0 && length(consumer2_prey) == 0
        return 0.0
    end

    intersection = length(intersect(consumer1_prey, consumer2_prey))
    union_size = length(union(consumer1_prey, consumer2_prey))

    return union_size > 0 ? intersection / union_size : 0.0
end

function find_similar_consumer(consumer_id::Int, params::FoodWebParameters)
    """
    Find the most similar consumer based on prey overlap (Gilljam approach).
    """
    consumer_prey = findall(params.adjacency[consumer_id, :])

    best_similarity = -1.0
    best_consumer = nothing

    for other_id in 1:params.S
        if other_id != consumer_id && !params.is_basal[other_id] && !params.is_extinct[other_id]
            other_prey = findall(params.adjacency[other_id, :])
            if length(other_prey) > 0
                similarity = calculate_jaccard_similarity(consumer_prey, other_prey)
                if similarity > best_similarity
                    best_similarity = similarity
                    best_consumer = other_id
                end
            end
        end
    end

    return best_consumer, best_similarity
end

function select_new_prey(consumer_id::Int, params::FoodWebParameters,
    N::Vector{Float64})
    """
    Select a new prey species for a consumer that needs to rewire.
    Uses Gilljam's similarity approach, falls back to random if needed.
    Respects trophic level and body mass constraints.
    """
    current_prey = findall(params.adjacency[consumer_id, :])
    available_prey = findall((1:params.S) .!= consumer_id .&&
                             N .> EXTINCTION_THRESHOLD .&&
                             .!params.is_extinct)

    # Remove already-consumed prey
    available_prey = setdiff(available_prey, current_prey)

    if length(available_prey) == 0
        return nothing, nothing, false
    end

    # Apply trophic level constraint: can only eat species at LOWER trophic level
    if USE_TROPHIC_CONSTRAINTS
        consumer_trophic = params.trophic_level[consumer_id]
        available_prey = filter(j -> params.trophic_level[j] < consumer_trophic,
            available_prey)
    end

    # Apply body mass constraint: predator/prey mass ratio must be in range
    if USE_BODY_MASS_CONSTRAINTS
        consumer_mass = params.body_mass[consumer_id]
        available_prey = filter(j -> begin
                mass_ratio = consumer_mass / params.body_mass[j]
                mass_ratio >= MIN_PREDATOR_PREY_MASS_RATIO &&
                    mass_ratio <= MAX_PREDATOR_PREY_MASS_RATIO
            end, available_prey)
    end

    if length(available_prey) == 0
        return nothing, nothing, false
    end

    # Try similarity-based selection
    similar_consumer, similarity = find_similar_consumer(consumer_id, params)

    if !isnothing(similar_consumer) && similarity > 0
        similar_prey = findall(params.adjacency[similar_consumer, :])
        candidate_prey = intersect(available_prey, similar_prey)

        if length(candidate_prey) > 0
            return rand(candidate_prey), similar_consumer, false
        end
    end

    # Fall back to random selection from available prey
    return rand(available_prey), nothing, true
end

function rewire!(consumer_id::Int, new_prey_id::Int,
    params::FoodWebParameters, state::AdaptiveState,
    similar_consumer::Union{Int,Nothing})
    """
    Create a new feeding link with rewiring cost applied.
    """
    # Add new link
    params.adjacency[consumer_id, new_prey_id] = true

    # Apply rewiring cost to attack rate
    if !isnothing(similar_consumer)
        # Use similar consumer's attack rate as base
        base_rate = params.attack_rates[similar_consumer, new_prey_id]
    else
        # Use a default rate
        base_rate = mean(filter(x -> x > 0, params.attack_rates[consumer_id, :]))
        if isnan(base_rate) || base_rate == 0
            base_rate = 1.0
        end
    end

    params.attack_rates[consumer_id, new_prey_id] = base_rate * (1 - REWIRING_COST)

    # Initialize small foraging effort and renormalize
    current_prey = findall(params.adjacency[consumer_id, :])
    n_prey = length(current_prey)

    if n_prey > 0
        initial_effort = 0.1 / n_prey
        state.foraging_efforts[consumer_id, new_prey_id] = initial_effort

        # Renormalize all efforts to sum to 1
        total_effort = sum(state.foraging_efforts[consumer_id, current_prey])
        if total_effort > 0
            state.foraging_efforts[consumer_id, current_prey] .*= (1 - initial_effort) / total_effort
        end
    end
end

function remove_prey_link!(consumer_id::Int, prey_id::Int,
    params::FoodWebParameters, state::AdaptiveState)
    """
    Remove a feeding link and renormalize foraging efforts.
    """
    params.adjacency[consumer_id, prey_id] = false
    params.attack_rates[consumer_id, prey_id] = 0.0
    state.foraging_efforts[consumer_id, prey_id] = 0.0

    # Renormalize remaining efforts
    current_prey = findall(params.adjacency[consumer_id, :])
    if length(current_prey) > 0
        total_effort = sum(state.foraging_efforts[consumer_id, current_prey])
        if total_effort > 0
            state.foraging_efforts[consumer_id, current_prey] ./= total_effort
        else
            state.foraging_efforts[consumer_id, current_prey] .= 1.0 / length(current_prey)
        end
    end
end

# ============================================================================
# INVASION FUNCTIONS
# ============================================================================

function add_invader!(params::FoodWebParameters, state::AdaptiveState,
    N::Vector{Float64}, t::Float64)
    """
    Add a new consumer species as an invader.
    Returns the invader ID if successful, nothing otherwise.
    """
    S_current = params.S

    # Find potential prey (living species)
    potential_prey = findall(N .> EXTINCTION_THRESHOLD)
    if length(potential_prey) == 0
        return nothing
    end

    # Create new species
    new_id = S_current + 1

    # Expand all arrays
    params.S = new_id
    push!(params.is_basal, false)
    push!(params.r, 0.0)
    push!(params.d, rand(Uniform(0.1, 0.3)))
    push!(params.K, 0.0)
    push!(params.can_rewire, rand() < FRACTION_CAN_REWIRE)
    push!(params.can_adapt, rand() < FRACTION_CAN_ADAPT)
    push!(params.is_extinct, false)

    # Expand matrices
    new_adjacency = zeros(Bool, new_id, new_id)
    new_adjacency[1:S_current, 1:S_current] = params.adjacency

    new_attack = zeros(new_id, new_id)
    new_attack[1:S_current, 1:S_current] = params.attack_rates

    new_conversion = zeros(new_id, new_id)
    new_conversion[1:S_current, 1:S_current] = params.conversion_eff

    new_handling = zeros(new_id, new_id)
    new_handling[1:S_current, 1:S_current] = params.handling_times

    new_q = ones(new_id, new_id)
    new_q[1:S_current, 1:S_current] = params.q

    # Give invader 1-3 prey randomly
    n_prey = rand(1:min(3, length(potential_prey)))
    invader_prey = sample(potential_prey, n_prey, replace=false)

    for prey_id in invader_prey
        new_adjacency[new_id, prey_id] = true
        new_attack[new_id, prey_id] = rand(Uniform(0.5, 1.5))
        new_conversion[new_id, prey_id] = 0.3
        if FUNCTIONAL_RESPONSE != :type_I
            new_handling[new_id, prey_id] = 0.2
        end
        if FUNCTIONAL_RESPONSE == :type_III
            new_q[new_id, prey_id] = 1.3
        end
    end

    params.adjacency = new_adjacency
    params.attack_rates = new_attack
    params.conversion_eff = new_conversion
    params.handling_times = new_handling
    params.q = new_q

    # Expand state matrices
    new_efforts = zeros(new_id, new_id)
    new_efforts[1:S_current, 1:S_current] = state.foraging_efforts
    new_efforts[new_id, invader_prey] .= 1.0 / n_prey
    state.foraging_efforts = new_efforts

    new_original = zeros(new_id, new_id)
    new_original[1:S_current, 1:S_current] = state.original_attack_rates
    new_original[new_id, :] = new_attack[new_id, :]
    state.original_attack_rates = new_original

    return new_id
end

# ============================================================================
# CALLBACKS
# ============================================================================

function create_extinction_callback(params, state, event_log)
    """
    Detect extinctions and trigger rewiring if enabled.
    Uses PeriodicCallback to check for extinctions at regular intervals.
    """
    function check_extinctions!(integrator)
        N = integrator.u[1:params.S]

        # Find ALL newly extinct species
        for species_id in 1:params.S
            if N[species_id] < EXTINCTION_THRESHOLD && !params.is_extinct[species_id]
                # Mark as extinct
                params.is_extinct[species_id] = true
                integrator.u[species_id] = 0.0

                # Log extinction
                push!(event_log, (time=integrator.t, event_type="EXTINCTION",
                    species_id=species_id, details=""))

                # Check which consumers need to rewire
                if ENABLE_EXTINCTION_REWIRING
                    consumers_affected = findall(params.adjacency[:, species_id])

                    for consumer_id in consumers_affected
                        if params.is_extinct[consumer_id]
                            # Consumer already extinct
                            continue
                        end

                        if !params.can_rewire[consumer_id]
                            # Consumer cannot rewire
                            push!(event_log, (time=integrator.t,
                                event_type="NO_REWIRE_ABILITY",
                                species_id=consumer_id,
                                details="lost prey $species_id but cannot rewire"))
                            continue
                        end

                        # Remove extinct prey link
                        remove_prey_link!(consumer_id, species_id, params, state)

                        # Check if consumer has any remaining prey
                        remaining_prey = findall(params.adjacency[consumer_id, :] .&&
                                                 .!params.is_extinct .&&
                                                 (N .> EXTINCTION_THRESHOLD))

                        if length(remaining_prey) == 0
                            # Check if consumer has exceeded max rewiring limit
                            if params.rewiring_count[consumer_id] >= MAX_REWIRING_PER_SPECIES
                                # Hit rewiring limit, consumer will go extinct
                                push!(event_log, (time=integrator.t,
                                    event_type="STARVATION",
                                    species_id=consumer_id,
                                    details="hit max rewiring limit ($MAX_REWIRING_PER_SPECIES)"))
                                params.is_extinct[consumer_id] = true
                                integrator.u[consumer_id] = 0.0
                                continue
                            end

                            # Must rewire to survive
                            new_prey, similar_consumer, is_random =
                                select_new_prey(consumer_id, params, N)

                            if !isnothing(new_prey)
                                rewire!(consumer_id, new_prey, params, state, similar_consumer)

                                # Update rewiring tracking (no cooldown for forced extinction rewiring)
                                params.last_rewiring_time[consumer_id] = integrator.t
                                params.rewiring_count[consumer_id] += 1

                                # Log rewiring event
                                event = RewiringEvent(integrator.t, EXTINCTION_FORCED,
                                    consumer_id, species_id, new_prey,
                                    similar_consumer, is_random, nothing)
                                push!(event_log, (time=event.time,
                                    event_type="REWIRING_EXTINCTION",
                                    species_id=consumer_id,
                                    details="prey=$species_id -> $new_prey, similar=$(similar_consumer), random=$is_random"))

                                # Update foraging efforts in integrator state
                                a_start = params.S + 1
                                for ii in 1:params.S
                                    for jj in 1:params.S
                                        integrator.u[a_start+(ii-1)*params.S+jj-1] =
                                            state.foraging_efforts[ii, jj]
                                    end
                                end
                            else
                                # No prey available, consumer will go extinct
                                push!(event_log, (time=integrator.t,
                                    event_type="STARVATION",
                                    species_id=consumer_id,
                                    details="no available prey"))
                                params.is_extinct[consumer_id] = true
                                integrator.u[consumer_id] = 0.0
                            end
                        else
                            # Consumer still has other prey, will reallocate efforts via Eq. 2
                            push!(event_log, (time=integrator.t,
                                event_type="PREY_LOST",
                                species_id=consumer_id,
                                details="lost prey $species_id, still has $(length(remaining_prey)) other prey"))
                        end
                    end
                end
            end
        end
    end

    # Check for extinctions every SAVE_INTERVAL time units
    PeriodicCallback(check_extinctions!, SAVE_INTERVAL,
        save_positions=(false, false))
end

function create_energy_callback(params, state, event_log)
    """
    Check for energy-based rewiring periodically.
    """
    function affect!(integrator)
        if !ENABLE_ENERGY_REWIRING
            return
        end

        N = integrator.u[1:params.S]
        a_start = params.S + 1

        # Reconstruct efforts matrix
        a = zeros(params.S, params.S)
        for i in 1:params.S
            for j in 1:params.S
                a[i, j] = integrator.u[a_start+(i-1)*params.S+j-1]
            end
        end

        for consumer_id in 1:params.S
            if params.is_basal[consumer_id] || params.is_extinct[consumer_id] ||
               N[consumer_id] < EXTINCTION_THRESHOLD || !params.can_rewire[consumer_id]
                continue
            end

            # Calculate total energy intake
            total_intake = 0.0
            prey_indices = findall(params.adjacency[consumer_id, :] .&&
                                   .!params.is_extinct .&&
                                   (N .> EXTINCTION_THRESHOLD))

            for prey_id in prey_indices
                prey_densities = [N[k] for k in 1:params.S]
                foraging_efforts_i = [a[consumer_id, k] for k in 1:params.S]
                attack_rates_i = [params.attack_rates[consumer_id, k] for k in 1:params.S]
                handling_times_i = [params.handling_times[consumer_id, k] for k in 1:params.S]

                FR = functional_response(N[prey_id],
                    params.attack_rates[consumer_id, prey_id],
                    params.handling_times[consumer_id, prey_id],
                    params.q[consumer_id, prey_id],
                    consumer_id, prey_densities,
                    foraging_efforts_i, attack_rates_i,
                    handling_times_i)

                total_intake += params.conversion_eff[consumer_id, prey_id] *
                                a[consumer_id, prey_id] * FR * N[prey_id]
            end

            # Check if intake falls below threshold
            maintenance_cost = params.d[consumer_id] * N[consumer_id]
            threshold = ENERGY_THRESHOLD_FACTOR * maintenance_cost

            if total_intake < threshold && length(prey_indices) > 0
                # Check rewiring constraints
                time_since_last = integrator.t - params.last_rewiring_time[consumer_id]
                can_rewire_now = (time_since_last >= REWIRING_COOLDOWN ||
                                  params.last_rewiring_time[consumer_id] == -Inf) &&
                                 (params.rewiring_count[consumer_id] < MAX_REWIRING_PER_SPECIES)

                if !can_rewire_now
                    continue  # Skip this consumer
                end

                # Try to find new prey
                new_prey, similar_consumer, is_random =
                    select_new_prey(consumer_id, params, N)

                if !isnothing(new_prey)
                    # CONDITIONAL: ADD vs REPLACE based on current diet breadth
                    n_current_prey = length(prey_indices)

                    if n_current_prey <= SPECIALIST_MAX_PREY
                        # SPECIALIST IN CRISIS: ADD supplemental prey to expand niche
                        rewire!(consumer_id, new_prey, params, state, similar_consumer)

                        # Update rewiring tracking
                        params.last_rewiring_time[consumer_id] = integrator.t
                        params.rewiring_count[consumer_id] += 1

                        push!(event_log, (time=integrator.t,
                            event_type="REWIRING_ENERGY",
                            species_id=consumer_id,
                            details="ADDED prey $new_prey (was specialist with $n_current_prey prey), intake_ratio=$(round(total_intake/max(maintenance_cost, 1e-10), digits=3)), similar=$similar_consumer, random=$is_random"))

                    else
                        # GENERALIST STRUGGLING: REPLACE least profitable prey to optimize
                        profitabilities = Dict{Int,Float64}()
                        for prey_id in prey_indices
                            profit = params.conversion_eff[consumer_id, prey_id] *
                                     params.attack_rates[consumer_id, prey_id] *
                                     N[prey_id]
                            profitabilities[prey_id] = profit
                        end

                        old_prey = argmin(profitabilities)
                        old_profit = profitabilities[old_prey]

                        if new_prey != old_prey
                            remove_prey_link!(consumer_id, old_prey, params, state)
                            rewire!(consumer_id, new_prey, params, state, similar_consumer)

                            # Update rewiring tracking
                            params.last_rewiring_time[consumer_id] = integrator.t
                            params.rewiring_count[consumer_id] += 1

                            push!(event_log, (time=integrator.t,
                                event_type="REWIRING_ENERGY",
                                species_id=consumer_id,
                                details="REPLACED prey $old_prey (profit=$(round(old_profit, digits=4))) -> $new_prey (had $n_current_prey prey), intake_ratio=$(round(total_intake/max(maintenance_cost, 1e-10), digits=3)), similar=$similar_consumer, random=$is_random"))
                        end
                    end

                    # Update foraging efforts
                    for ii in 1:params.S
                        for jj in 1:params.S
                            integrator.u[a_start+(ii-1)*params.S+jj-1] =
                                state.foraging_efforts[ii, jj]
                        end
                    end
                end
            end
        end
    end

    PeriodicCallback(affect!, ENERGY_REWIRING_INTERVAL)
end

function create_invasion_callback(params, state, event_log)
    """
    Periodically check for species invasions.
    """
    function affect!(integrator)
        if !ENABLE_INVASION || rand() > INVASION_RATE * INVASION_CHECK_INTERVAL
            return
        end

        N = integrator.u[1:params.S]
        S_before = params.S

        # Try to add invader
        invader_id = add_invader!(params, state, N, integrator.t)

        if !isnothing(invader_id)
            # Expand integrator state
            S_new = params.S
            new_u = zeros(S_new + S_new^2)

            # Copy populations
            new_u[1:S_before] = integrator.u[1:S_before]
            new_u[invader_id] = rand(Uniform(0.1, 1.0))  # Initial invader density

            # Copy foraging efforts
            for i in 1:S_before
                for j in 1:S_before
                    new_u[S_new+(i-1)*S_new+j] = integrator.u[S_before+(i-1)*S_before+j]
                end
            end

            # Set invader foraging efforts
            for j in 1:S_new
                new_u[S_new+(invader_id-1)*S_new+j] = state.foraging_efforts[invader_id, j]
            end

            # Resize integrator
            resize!(integrator, length(new_u))
            integrator.u .= new_u

            # Log invasion
            push!(event_log, (time=integrator.t, event_type="INVASION",
                species_id=invader_id,
                details="new consumer with $(sum(params.adjacency[invader_id, :])) prey"))

            println("  [t=$(round(integrator.t, digits=1))] Species $invader_id invaded!")
        end
    end

    PeriodicCallback(affect!, INVASION_CHECK_INTERVAL)
end

function create_stochasticity_callback(params)
    """
    Periodically perturb growth rates of basal species.
    """
    function affect!(integrator)
        if !STOCHASTICITY_ENABLED
            return
        end

        for i in 1:params.n_basal
            perturbation = 1.0 + STOCHASTICITY_INTENSITY * (2 * rand() - 1)
            params.r[i] *= perturbation
            params.r[i] = clamp(params.r[i], 0.1, 2.0)
        end
    end

    PeriodicCallback(affect!, STOCHASTICITY_INTERVAL)
end

function create_dynamic_replacement_callback(params, state, event_log)
    """
    Dynamic replacement: replace least profitable links with new ones.
    Each consumer has probability DYNAMIC_REPLACEMENT_RATE * dt of replacing a link.
    Follows Kondoh logic: drops least profitable prey (min e_ij * f_ij * X_j).
    """
    function affect!(integrator)
        if !ENABLE_DYNAMIC_REPLACEMENT
            return
        end

        N = integrator.u[1:params.S]
        a_start = params.S + 1

        for consumer_id in 1:params.S
            if params.is_basal[consumer_id] || params.is_extinct[consumer_id] ||
               N[consumer_id] < EXTINCTION_THRESHOLD || !params.can_rewire[consumer_id]
                continue
            end

            # Check if this consumer experiences a replacement event
            if rand() < DYNAMIC_REPLACEMENT_RATE
                # Check cooldown constraint (limit is checked globally via rewiring_count)
                time_since_last = integrator.t - params.last_rewiring_time[consumer_id]
                can_rewire_now = (time_since_last >= REWIRING_COOLDOWN ||
                                  params.last_rewiring_time[consumer_id] == -Inf) &&
                                 (params.rewiring_count[consumer_id] < MAX_REWIRING_PER_SPECIES)

                if !can_rewire_now
                    continue  # Skip this consumer
                end

                current_prey = findall(params.adjacency[consumer_id, :] .&&
                                       .!params.is_extinct .&&
                                       (N .> EXTINCTION_THRESHOLD))

                if length(current_prey) == 0
                    continue
                end

                # Calculate profitability (e_ij * f_ij * X_j) for each current prey
                profitabilities = Dict{Int,Float64}()
                for prey_id in current_prey
                    profit = params.conversion_eff[consumer_id, prey_id] *
                             params.attack_rates[consumer_id, prey_id] *
                             N[prey_id]
                    profitabilities[prey_id] = profit
                end

                # Select LEAST profitable prey to replace
                old_prey = argmin(profitabilities)

                # Try to find a new prey (that isn't already being consumed)
                new_prey, similar_consumer, is_random = select_new_prey(consumer_id, params, N)

                if !isnothing(new_prey) && new_prey != old_prey
                    # Calculate profitability of new prey
                    new_profit = params.conversion_eff[consumer_id, new_prey] *
                                 params.attack_rates[consumer_id, new_prey] *
                                 N[new_prey]

                    # Accept if new prey is within 10% of old prey profitability
                    if new_profit > profitabilities[old_prey] * 0.9
                        # Remove old link
                        remove_prey_link!(consumer_id, old_prey, params, state)

                        # Add new link
                        rewire!(consumer_id, new_prey, params, state, similar_consumer)

                        # Update rewiring tracking
                        params.last_rewiring_time[consumer_id] = integrator.t
                        params.rewiring_count[consumer_id] += 1

                        # Log the replacement event
                        push!(event_log, (time=integrator.t,
                            event_type="REWIRING_DYNAMIC",
                            species_id=consumer_id,
                            details="replaced prey $old_prey (profit=$(round(profitabilities[old_prey], digits=4))) -> $new_prey (profit=$(round(new_profit, digits=4))), similar=$similar_consumer"))

                        # Update foraging efforts in integrator state
                        for ii in 1:params.S
                            for jj in 1:params.S
                                integrator.u[a_start+(ii-1)*params.S+jj-1] =
                                    state.foraging_efforts[ii, jj]
                            end
                        end
                    end
                end
            end
        end
    end

    PeriodicCallback(affect!, 1.0)  # Check every time unit
end

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

function run_burnin(params::FoodWebParameters, state::AdaptiveState;
    burnin_time::Float64=BURNIN_TIME)
    """
    Run deterministic burn-in to find reasonable initial conditions.
    Viable initial conditions = all species have density > EXTINCTION_THRESHOLD after equilibration.
    """
    N0 = rand(Uniform(0.5, 3.0), params.S)  # Lower initial densities
    a0 = vec(state.foraging_efforts)
    u0 = vcat(N0, a0)

    # Run without stochasticity to equilibrate
    # Use longer time and more relaxed tolerances for lower mortality rates
    prob = ODEProblem(foodweb_dynamics!, u0, (0.0, burnin_time * 2.0), (params, state))
    sol = solve(prob, Rosenbrock23(autodiff=false), saveat=burnin_time * 2.0,
        abstol=1e-6, reltol=1e-4, maxiters=1e6,
        isoutofdomain=(u, p, t) -> any(u[1:params.S] .< -1e-10))

    if sol.retcode != :Success
        return nothing, false
    end

    # Viable = all species survived burn-in (density > threshold)
    N_final = sol.u[end][1:params.S]
    all_alive = all(N_final .> EXTINCTION_THRESHOLD)

    return sol.u[end], all_alive
end

function single_run(params::FoodWebParameters, state::AdaptiveState;
    u0=nothing, run_name::String="run", tspan_override=nothing)
    """
    Run a single simulation with all mechanisms enabled.
    """
    println("\n=== Running: $run_name ===")

    # Get initial condition
    if USE_RANDOM_INITIAL || !isnothing(u0)
        if isnothing(u0)
            N0 = rand(Uniform(1.0, 2.0), params.S)
            a0 = vec(state.foraging_efforts)
            u0 = vcat(N0, a0)
        end
        println("Using provided/random initial conditions")
    else
        println("Running burn-in...")
        u0, success = run_burnin(params, state)
        if !success || isnothing(u0)
            println("ERROR: Burn-in failed - could not find stable initial conditions")
            return nothing, nothing, nothing
        end
        println("Burn-in successful")
    end

    # Setup
    event_log = []
    tspan_sim = isnothing(tspan_override) ? SIMULATION_TIME : tspan_override
    tspan = (0.0, tspan_sim)

    # Store initial state at START of main simulation
    S_initial = params.S
    N_at_start = copy(u0[1:S_initial])  # Densities at start of main sim

    # Create callbacks
    extinction_cb = create_extinction_callback(params, state, event_log)
    energy_cb = create_energy_callback(params, state, event_log)
    dynamic_cb = create_dynamic_replacement_callback(params, state, event_log)
    stochasticity_cb = create_stochasticity_callback(params)
    invasion_cb = create_invasion_callback(params, state, event_log)

    callback_set = CallbackSet(extinction_cb, energy_cb, dynamic_cb, stochasticity_cb, invasion_cb)

    # Solve
    println("Simulating dynamics...")
    prob = ODEProblem(foodweb_dynamics!, u0, tspan, (params, state))
    sol = solve(prob, Rosenbrock23(autodiff=false), callback=callback_set,
        saveat=SAVE_INTERVAL, abstol=1e-8, reltol=1e-6,
        isoutofdomain=(u, p, t) -> any(u[1:params.S] .< -1e-10))

    if sol.retcode != :Success
        println("WARNING: Solver did not finish successfully (status: $(sol.retcode))")
    end

    # Calculate metrics using INITIAL species count (as in Gilljam paper)
    N_final = sol.u[end][1:S_initial]  # Only look at initial species

    # Count extinctions: species alive at START but extinct at END
    initially_alive = N_at_start .> EXTINCTION_THRESHOLD
    finally_extinct = N_final .<= EXTINCTION_THRESHOLD
    n_extinctions = sum(initially_alive .& finally_extinct)

    # Survived: species alive at both start and end
    n_survived = sum(initially_alive .& .!finally_extinct)

    # Initial species count: all species alive at start
    n_initial_species = sum(initially_alive)

    # Persistence = fraction of INITIAL (alive) species still alive
    persistence = n_survived / n_initial_species

    # Count rewiring events by type
    n_rewiring_extinction = count(e -> e.event_type == "REWIRING_EXTINCTION", event_log)
    n_rewiring_energy = count(e -> e.event_type == "REWIRING_ENERGY", event_log)
    n_rewiring_dynamic = count(e -> e.event_type == "REWIRING_DYNAMIC", event_log)
    n_rewiring_total = n_rewiring_extinction + n_rewiring_energy + n_rewiring_dynamic
    n_invasions = count(e -> e.event_type == "INVASION", event_log)

    # Summary
    summary = Dict(
        "run_name" => run_name,
        "initial_species" => n_initial_species,
        "final_species" => n_survived,
        "species_at_end" => params.S,  # May include invaders
        "n_extinctions" => n_extinctions,
        "persistence" => persistence,
        "n_rewiring_total" => n_rewiring_total,
        "n_rewiring_extinction" => n_rewiring_extinction,
        "n_rewiring_energy" => n_rewiring_energy,
        "n_rewiring_dynamic" => n_rewiring_dynamic,
        "n_invasions" => n_invasions,
        "solver_status" => string(sol.retcode),
        "adaptation_rate" => ADAPTATION_RATE,
        "rewiring_cost" => REWIRING_COST
    )

    println("\n--- Summary ---")
    println("Initial species: $n_initial_species")
    println("Species survived: $n_survived")
    println("Total extinctions: $n_extinctions")
    println("Persistence: $(round(persistence, digits=3))")
    if n_rewiring_total > 0
        rewiring_parts = String[]
        if n_rewiring_extinction > 0
            push!(rewiring_parts, "$n_rewiring_extinction extinction")
        end
        if n_rewiring_energy > 0
            push!(rewiring_parts, "$n_rewiring_energy energy")
        end
        if n_rewiring_dynamic > 0
            push!(rewiring_parts, "$n_rewiring_dynamic dynamic")
        end
        println("Rewiring events: $n_rewiring_total ($(join(rewiring_parts, ", ")))")
    else
        println("Rewiring events: 0")
    end
    if n_invasions > 0
        println("Invasions: $n_invasions")
    end
    println("Final S (with invaders): $(params.S)")

    # Save results
    output_dir = joinpath("results", "single_runs", run_name)
    mkpath(output_dir)

    # Save summary
    df_summary = DataFrame(summary)
    CSV.write(joinpath(output_dir, "summary.csv"), df_summary)

    # Save event log
    if length(event_log) > 0
        df_events = DataFrame(event_log)
        CSV.write(joinpath(output_dir, "events.csv"), df_events)
    end

    # Save time series
    times = sol.t
    N_series = hcat([sol.u[i][1:S_initial] for i in 1:length(sol)]...)'  # Only initial species
    df_timeseries = DataFrame(N_series, :auto)
    rename!(df_timeseries, [Symbol("sp_$i") for i in 1:S_initial])
    df_timeseries[!, :time] = times
    CSV.write(joinpath(output_dir, "timeseries.csv"), df_timeseries)

    # Create and save timeseries plot
    plot_timeseries(sol, params, joinpath(output_dir, "timeseries.png"),
        S_initial=S_initial)

    # Create and save species-level summary
    df_species = create_species_summary(params, sol, event_log, N_at_start, S_initial)
    CSV.write(joinpath(output_dir, "species_summary.csv"), df_species)

    println("Results saved to: $output_dir")
    println("  - summary.csv: run-level summary")
    println("  - events.csv: detailed event log")
    println("  - timeseries.csv: population time series")
    println("  - timeseries.png: population dynamics plot")
    println("  - species_summary.csv: per-species metrics")

    return summary, sol, event_log
end

function parameter_sweep(param_ranges::Dict; n_replicates::Int=10,
    web_type::Symbol=:cascade, show_progress::Bool=true)
    """
    Run parameter sweep across specified ranges.
    """
    println("\n=== Parameter Sweep ===")
    println("Replicates per combination: $n_replicates")
    println("Parameter ranges:")
    for (param, values) in param_ranges
        println("  $param: $values")
    end

    # Generate all parameter combinations
    param_names = collect(keys(param_ranges))
    param_values = collect(values(param_ranges))
    combinations = vec(collect(Iterators.product(param_values...)))

    results = []
    total_runs = length(combinations) * n_replicates

    # Create progress bar
    if show_progress
        p = Progress(total_runs, desc="Parameter sweep: ")
    end

    for combo in combinations
        params_dict = Dict(zip(param_names, combo))
        S = params_dict[:S]
        C = params_dict[:C]

        for rep in 1:n_replicates
            # Generate food web
            if web_type == :cascade
                adjacency = generate_cascade_foodweb(S, C)
            else
                adjacency = generate_random_foodweb(S, C)
            end

            params = initialize_parameters(adjacency, model_type=web_type)
            state = initialize_foraging_efforts(params)

            # Run simulation with shorter time for sweeps
            summary, sol, events = single_run(params, state,
                run_name="sweep_$(combo)_rep$(rep)",
                tspan_override=SWEEP_SIMULATION_TIME)

            if !isnothing(summary)
                result = merge(params_dict, summary)
                result[:replicate] = rep
                push!(results, result)
            end

            if show_progress
                next!(p)
            end
        end
    end

    # Convert to DataFrame
    df_results = DataFrame(results)

    # Save results
    output_dir = joinpath("results", "sweeps")
    mkpath(output_dir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = joinpath(output_dir, "parameter_sweep_$timestamp.csv")
    CSV.write(filename, df_results)

    println("\nSweep complete! Results saved to: $filename")

    return df_results
end

# ============================================================================
# VISUALIZATION
# ============================================================================

function get_trophic_level(species_id::Int, adjacency::Matrix{Bool})
    """
    Calculate trophic level using AVERAGE prey TL: 1 for basal, mean(prey TLs) + 1 for consumers.
    This fractional approach accounts for omnivory and prevents unrealistic TL inflation.
    """
    S = size(adjacency, 1)
    levels = zeros(Float64, S)  # Float64 for fractional levels

    # Find basal species (no prey)
    for i in 1:S
        if sum(adjacency[i, :]) == 0
            levels[i] = 1.0
        end
    end

    # Iteratively assign levels using AVERAGE prey TL
    changed = true
    max_iter = S
    iter = 0
    while changed && iter < max_iter
        changed = false
        iter += 1
        for i in 1:S
            if levels[i] == 0.0
                prey = findall(adjacency[i, :])
                if length(prey) > 0 && all(levels[prey] .> 0)
                    levels[i] = mean(levels[prey]) + 1.0  # AVERAGE not MAXIMUM
                    changed = true
                end
            end
        end
    end

    return levels[species_id]
end

function get_trophic_positions(adjacency::Matrix{Bool}, is_basal::Vector{Bool})
    """
    Calculate x,y positions for trophic layout.
    Handles fractional trophic levels by rounding for layout purposes.
    """
    S = size(adjacency, 1)
    levels = [get_trophic_level(i, adjacency) for i in 1:S]

    # Round to nearest 0.5 for grouping (handles fractional TLs from omnivory)
    rounded_levels = round.(levels * 2) / 2
    unique_levels = sort(unique(rounded_levels))
    max_level = maximum(unique_levels)

    # Group species by rounded trophic level
    level_groups = [findall(rounded_levels .== l) for l in unique_levels]

    # Assign positions
    locs_x = zeros(S)
    locs_y = zeros(S)

    for (level_idx, group) in enumerate(level_groups)
        n_in_level = length(group)
        for (i, species_id) in enumerate(group)
            locs_x[species_id] = (i - 1) / max(n_in_level - 1, 1)  # Spread horizontally
            y_base = (level_idx - 1) / max(length(unique_levels) - 1, 1)  # Base vertical position
            locs_y[species_id] = y_base + rand(Uniform(-0.03, 0.03))  # Jitter to prevent overlap
        end
    end

    return locs_x, locs_y
end

function visualize_network(params::FoodWebParameters, state::AdaptiveState,
    N::Vector{Float64}; filename::String="network.png",
    extinct_species::Union{Vector{Bool},BitVector}=falses(params.S))
    """
    Visualize food web with trophic layout. Edges point from prey to predator.
    """
    S = length(N)

    # Create directed graph (edges FROM prey TO predator)
    g = SimpleDiGraph(S)
    for i in 1:S
        for j in 1:S
            if params.adjacency[i, j]  # i eats j
                add_edge!(g, j, i)  # Edge FROM prey j TO predator i
            end
        end
    end

    # Node colors: green=basal, blue=consumer, red=invader, gray=extinct
    node_colors = Vector{Colorant}(undef, S)
    for i in 1:S
        if extinct_species[i] || N[i] < EXTINCTION_THRESHOLD
            node_colors[i] = colorant"gray"
        elseif !(i in params.initial_species)
            node_colors[i] = colorant"red"  # Invader
        elseif params.is_basal[i]
            node_colors[i] = colorant"green"
        else
            node_colors[i] = colorant"blue"  # Consumer
        end
    end

    # Edge properties based on foraging effort
    edge_colors = colorant"black"
    edge_widths = []
    for e in edges(g)
        predator = dst(e)
        prey = src(e)
        effort = state.foraging_efforts[predator, prey]
        width = 0.5 + 2.5 * effort
        push!(edge_widths, width)
    end

    # Get trophic positions and flip so basal is at bottom
    locs_x, locs_y = get_trophic_positions(params.adjacency, params.is_basal)
    locs_y = 1.0 .- locs_y  # Flip: basal at bottom (y≈0), consumers at top (y≈1)

    # Node labels (species IDs)
    nodelabel = [extinct_species[i] || N[i] < EXTINCTION_THRESHOLD ? "✕$i" : "$i" for i in 1:S]

    # Node sizes proportional to log(body mass) for better visualization
    node_sizes = log10.(params.body_mass .+ 1) .* 0.5  # Scale factor 0.5 for reasonable display

    # Plot
    p = gplot(g,
        locs_x, locs_y,
        nodefillc=node_colors,
        nodesize=node_sizes,
        edgestrokec=edge_colors,
        edgelinewidth=edge_widths,
        arrowlengthfrac=0.05,
        nodelabel=nodelabel,
        nodelabeldist=0,
        nodelabelsize=6.0,
        background_color=colorant"white")

    # Save as PDF with smaller size
    if endswith(filename, ".png")
        filename = replace(filename, ".png" => ".pdf")
    end
    img = PDF(filename, 12cm, 12cm)
    draw(img, compose(context(), Compose.rectangle(), fill("white"), p))
    println("Network visualization saved to: $filename")
    println("  Green = basal, Blue = consumers, Red = invaders, Gray/✕ = extinct")
    println("  Edge width = foraging effort, Basal at bottom, Consumers at top")
    println("  Edge width = foraging effort")
end

function plot_timeseries(sol, params::FoodWebParameters, output_file::String;
    S_initial::Union{Int,Nothing}=nothing)
    """
    Plot population dynamics time series for all species.
    """
    S_plot = isnothing(S_initial) ? params.S : S_initial
    times = sol.t

    # Extract population densities (only initial species if specified)
    N_series = hcat([sol.u[i][1:S_plot] for i in 1:length(sol)]...)'

    # Use Juno theme
    theme(:juno)

    # Create plot
    p = plot(legend=:outerright, size=(1000, 600),
        xlabel="Time", ylabel="Density",
        title="Population Dynamics")

    # Plot each species
    for i in 1:S_plot
        species_label = if params.is_basal[i]
            "Basal $i"
        else
            "Consumer $i"
        end

        # Check if species went extinct
        if params.is_extinct[i]
            species_label *= " (✕)"
            plot!(p, times, N_series[:, i], label=species_label,
                linestyle=:dash, alpha=0.6)
        else
            plot!(p, times, N_series[:, i], label=species_label)
        end
    end

    savefig(p, output_file)
    println("Timeseries plot saved to: $output_file")

    return p
end

function create_species_summary(params::FoodWebParameters, sol, event_log,
    N_at_start::Vector{Float64}, S_initial::Int)
    """
    Create a detailed summary DataFrame for each species.
    """
    N_final = sol.u[end][1:S_initial]

    species_data = []

    for i in 1:S_initial
        # Count events involving this species
        n_extinction_rewiring = 0
        n_energy_rewiring = 0
        n_dynamic_rewiring = 0
        n_times_as_new_prey = 0
        n_times_as_old_prey = 0
        extinction_time = NaN

        for event in event_log
            if event.event_type == "EXTINCTION" && event.species_id == i
                extinction_time = event.time
            elseif event.event_type == "REWIRING_EXTINCTION" && event.species_id == i
                n_extinction_rewiring += 1
            elseif event.event_type == "REWIRING_ENERGY" && event.species_id == i
                n_energy_rewiring += 1
            elseif event.event_type == "REWIRING_DYNAMIC" && event.species_id == i
                n_dynamic_rewiring += 1
            end

            # Check if this species was involved as prey in rewiring
            if haskey(event, :details) && occursin("new_prey=$i", string(event.details))
                n_times_as_new_prey += 1
            end
            if haskey(event, :details) && occursin("old_prey=$i", string(event.details))
                n_times_as_old_prey += 1
            end
        end

        # Calculate initial and final number of prey and predators
        initial_n_prey = sum(params.adjacency[i, 1:S_initial])
        initial_n_predators = sum(params.adjacency[1:S_initial, i])

        push!(species_data, (
            species_id=i,
            is_basal=params.is_basal[i],
            can_rewire=params.can_rewire[i],
            can_adapt=params.can_adapt[i],
            trophic_level=params.trophic_level[i],
            body_mass=params.body_mass[i],
            initial_abundance=N_at_start[i],
            final_abundance=N_final[i],
            extinct=params.is_extinct[i],
            extinction_time=extinction_time,
            initial_n_prey=initial_n_prey,
            initial_n_predators=initial_n_predators,
            n_extinction_rewiring_events=n_extinction_rewiring,
            n_energy_rewiring_events=n_energy_rewiring,
            n_dynamic_rewiring_events=n_dynamic_rewiring,
            n_total_rewiring_events=n_extinction_rewiring + n_energy_rewiring + n_dynamic_rewiring,
            n_times_targeted_as_new_prey=n_times_as_new_prey,
            n_times_lost_as_old_prey=n_times_as_old_prey,
            growth_or_mortality_rate=params.is_basal[i] ? params.r[i] : -params.d[i],
            carrying_capacity=params.is_basal[i] ? params.K[i] : NaN
        ))
    end

    return DataFrame(species_data)
end

function plot_persistence_heatmap(df_results::DataFrame, param1::Symbol,
    param2::Symbol, output_file::String)
    """
    Create heatmap of persistence vs two parameters.
    """
    param1_vals = sort(unique(df_results[!, param1]))
    param2_vals = sort(unique(df_results[!, param2]))

    persistence_matrix = zeros(length(param2_vals), length(param1_vals))

    for (i, p2) in enumerate(param2_vals)
        for (j, p1) in enumerate(param1_vals)
            rows = (df_results[!, param1] .== p1) .& (df_results[!, param2] .== p2)
            if any(rows)
                persistence_matrix[i, j] = mean(df_results[rows, :persistence])
            end
        end
    end

    p = heatmap(param1_vals, param2_vals, persistence_matrix,
        xlabel=string(param1), ylabel=string(param2),
        title="Species Persistence",
        color=:viridis, clims=(0, 1))

    savefig(p, output_file)
    println("Heatmap saved to: $output_file")
end

# ============================================================================
# MAIN EXECUTION EXAMPLES
# ============================================================================

function example_single_run()
    """
    Example: Run a single detailed simulation.
    """
    println("=== Single Run Example ===\n")

    max_attempts = 10
    for attempt in 1:max_attempts
        println("Attempt $attempt to generate viable food web...")

        S = 25
        C = 0.2
        adjacency = generate_cascade_foodweb(S, C)

        # Initialize
        params = initialize_parameters(adjacency, model_type=:cascade)
        state = initialize_foraging_efforts(params)

        output_dir = joinpath("results", "single_runs", "example_cascade")
        mkpath(output_dir)

        if !USE_RANDOM_INITIAL
            u0, success = run_burnin(params, state)
            if !success || isnothing(u0)
                println("Burn-in failed, trying again...\n")
                continue
            end
            # Visualize AFTER burn-in with actual starting densities
            N_initial = u0[1:params.S]
        else
            u0 = nothing
            N_initial = rand(Uniform(0.5, 3.0), params.S)
        end

        # Store initial adjacency and state for comparison
        initial_adjacency = copy(params.adjacency)
        initial_efforts = copy(state.foraging_efforts)

        # Visualize initial network with actual starting conditions
        visualize_network(params, state, N_initial,
            filename=joinpath(output_dir, "graph_initial.pdf"))

        println("Successfully found viable initial conditions!\n")
        summary, sol, events = single_run(params, state, u0=u0, run_name="example_cascade")

        if !isnothing(summary)
            # Visualize final network showing changed adjacency and efforts
            S_initial = length(params.initial_species)
            N_final = sol.u[end][1:S_initial]

            # Extract final foraging efforts from solution
            a_final = reshape(sol.u[end][(S_initial+1):(S_initial+S_initial^2)], S_initial, S_initial)

            # Create a view of final state (adjacency may have changed due to rewiring)
            final_params = FoodWebParameters(
                S_initial, params.n_basal,
                params.r[1:S_initial], params.d[1:S_initial], params.K[1:S_initial],
                params.attack_rates[1:S_initial, 1:S_initial],
                params.conversion_eff[1:S_initial, 1:S_initial],
                params.handling_times[1:S_initial, 1:S_initial],
                params.q[1:S_initial, 1:S_initial],
                params.adjacency[1:S_initial, 1:S_initial],  # May have rewiring changes
                params.is_basal[1:S_initial],
                params.can_rewire[1:S_initial],
                params.can_adapt[1:S_initial],
                params.is_extinct[1:S_initial],
                params.initial_species,
                params.trophic_level[1:S_initial],
                params.body_mass[1:S_initial],
                params.last_rewiring_time[1:S_initial],  # ADD THIS
                params.rewiring_count[1:S_initial]       # AND THIS
            )

            final_state = AdaptiveState(
                a_final,
                state.original_attack_rates[1:S_initial, 1:S_initial]
            )

            visualize_network(final_params, final_state, N_final,
                filename=joinpath(output_dir, "graph_final.pdf"),
                extinct_species=Vector{Bool}(params.is_extinct[1:S_initial]))
        end

        return summary, sol, events
    end

    println("ERROR: Could not generate a viable food web after $max_attempts attempts.")
    return nothing, nothing, nothing
end

function example_parameter_sweep()
    """
    Example: Run parameter sweep for persistence heatmap.
    """
    println("=== Parameter Sweep Example ===\n")

    param_ranges = Dict(
        :S => [10, 15, 20],
        :C => [0.1, 0.15, 0.2, 0.25]
    )

    results = parameter_sweep(param_ranges, n_replicates=10, web_type=:cascade)

    output_dir = joinpath("results", "sweeps")
    plot_persistence_heatmap(results, :S, :C,
        joinpath(output_dir, "persistence_S_vs_C.png"))

    return results
end

function example_rewiring_comparison()
    """
    Example: Compare scenarios with/without different rewiring mechanisms.
    """
    println("=== Rewiring Comparison Example ===\n")

    scenarios = [
        ("no_rewiring", false, false, false),
        ("extinction_only", true, false, false),
        ("extinction_energy", true, true, false),
        ("all_rewiring", true, true, true)
    ]

    S = 12
    C = 0.15

    results_summary = DataFrame(
        scenario=String[],
        persistence=Float64[],
        extinctions=Int[],
        rewiring_events=Int[]
    )

    for (name, ext, energy, dynamic) in scenarios
        global ENABLE_EXTINCTION_REWIRING = ext
        global ENABLE_ENERGY_REWIRING = energy
        global ENABLE_DYNAMIC_REPLACEMENT = dynamic

        println("\nTesting scenario: $name")

        adjacency = generate_cascade_foodweb(S, C)
        params = initialize_parameters(adjacency, model_type=:cascade)
        state = initialize_foraging_efforts(params)

        summary, sol, events = single_run(params, state, run_name=name)

        if !isnothing(summary)
            push!(results_summary, (
                name,
                summary["persistence"],
                summary["n_extinctions"],
                summary["n_rewiring_total"]
            ))
        end
    end

    output_dir = joinpath("results", "comparisons")
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "rewiring_comparison.csv"), results_summary)

    println("\n=== Comparison Summary ===")
    println(results_summary)

    return results_summary
end

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

println("""
╔════════════════════════════════════════════════════════════════╗
║  Adaptive Food Web Dynamics with Rewiring Simulation           ║
║  Based on Kondoh (2003) + Gilljam et al. (2015)                ║
╚════════════════════════════════════════════════════════════════╝

Current Configuration:
  - Functional Response: $FUNCTIONAL_RESPONSE
  - Adaptation Rate (G): $ADAPTATION_RATE (0 = no adaptation)
  - Extinction Rewiring: $ENABLE_EXTINCTION_REWIRING
  - Energy Rewiring: $ENABLE_ENERGY_REWIRING (threshold = $ENERGY_THRESHOLD_FACTOR)
  - Dynamic Replacement: $ENABLE_DYNAMIC_REPLACEMENT
  - Invasions: $ENABLE_INVASION
  - Fraction Can Rewire: $FRACTION_CAN_REWIRE
  - Fraction Can Adapt: $FRACTION_CAN_ADAPT
  - Diet Breadth Control:
    • Specialist max prey: $SPECIALIST_MAX_PREY (energy ADDS if ≤ this)
    • Generalist max prey: $GENERALIST_MAX_PREY (stops adding at this limit)
  - Use Random Initial: $USE_RANDOM_INITIAL
  - Threads Available: $(Threads.nthreads())

Available Functions:
  - example_single_run()          : Run single detailed simulation
  - example_parameter_sweep()     : Parameter sweep for heatmaps
  - example_rewiring_comparison() : Compare rewiring scenarios

""")

# Uncomment to run examples:
summary, sol, events = example_single_run()
# results = example_parameter_sweep()
# comparison = example_rewiring_comparison()