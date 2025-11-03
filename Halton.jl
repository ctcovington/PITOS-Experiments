using HypothesisTests
using Distributions
using PyPlot
using SpecialFunctions
using StatsBase
using Random
using Printf
using DelimitedFiles
using Combinatorics
using StatsBase

cauchy_combination(pvalues) = ccdf(Cauchy(), mean(tan.(pi*(0.5 .- pvalues))))

"""
    halton_1d(i::Int, b::Int)

Calculates the i-th element of a 1D Halton sequence with a given base `b`.

The Halton sequence is a deterministic, low-discrepancy sequence. The calculation
involves representing the index `i` in the given base `b`, reversing the digits,
and interpreting the result as a fraction.

# Arguments
- `i::Int`: The index of the element to generate (starting from 1).
- `b::Int`: The base for the sequence (must be a prime number for best results).

# Returns
- `Float64`: The i-th value in the Halton sequence for base `b`.
"""
function halton_1d(i::Int, b::Int)
    result = 0.0
    f = 1.0
    
    # This loop effectively performs the base conversion and reversal.
    while i > 0
        f /= b
        result += f * (i % b)
        i = div(i, b) # Integer division
    end
    
    return result
end

"""
    generate_rounded_halton_sequence(N::Int, n::Int)

Generates `N` points of a 2D Halton sequence and rounds each coordinate
to the nearest value of the form `k/n`.

This function uses the first two prime numbers (2 and 3) as bases for the
x and y coordinates, respectively, which is standard practice for 2D Halton sequences.

# Arguments
- `N::Int`: The total number of points to generate.
- `n::Int`: The fixed denominator for rounding. The grid size is determined by `n`.

# Returns
- `Vector{Tuple{Int, Int}}`: A vector of `N` rounded 2D points.
"""
function generate_rounded_halton_sequence(N::Int, n::Int)
    if n <= 0
        error("Denominator n must be a positive integer.")
    end

    # Standard prime bases for a 2D Halton sequence
    base_x = 2
    base_y = 3

    # Initialize a vector to store the generated points
    indices = Vector{Tuple{Int, Int}}()
    sizehint!(indices, N) # Pre-allocate memory for efficiency

    i = 1;
    halton_index = 1;
    while i <= N
        # 1. Generate the raw Halton coordinates for the i-th point
        x_raw = halton_1d(halton_index, base_x)
        y_raw = halton_1d(halton_index, base_y)

        # 2. Round each coordinate to the closest k/n.
        # This is equivalent to rounding (coord * n) to the nearest integer k,
        # and then calculating k/n.
        x_rounded = round(x_raw * n) / n
        y_rounded = round(y_raw * n) / n

        # 3. Calculate indices pertaining to each point
        x_index = round(Int, x_rounded * n)
        y_index = round(Int, y_rounded * n)
        if x_index != y_index
            push!(indices, (x_index, y_index))
            i += 1
        end
        halton_index += 1
    end

    return indices
end

function generate_extended_halton_sequence(N::Int, n::Int)
    halton_sequence = generate_rounded_halton_sequence(N, n)
    marginals = [(0, i) for i = 1:n]
    adjacent_pairs = [(i, i+1) for i = 1:n-1]
    rev_marginals = [(i, 0) for i = 1:n]
    rev_adjacent_pairs = [(i+1, i) for i = 1:n-1]
    extended_halton_sequence = vcat(halton_sequence, marginals, adjacent_pairs, rev_marginals, rev_adjacent_pairs)
    sampling_weights = calculate_scaled_weights(extended_halton_sequence, n)
    sampling_probabilities = pweights(sampling_weights)
    return (extended_halton_sequence, sampling_probabilities)
end

function generate_raw_halton_sequence(N::Int, n::Int)
    if n <= 0
        error("Denominator n must be a positive integer.")
    end

    # Standard prime bases for a 2D Halton sequence
    base_x = 2
    base_y = 3

    # Initialize a vector to store the generated points
    halton_sequence = Vector{Tuple{Float64, Float64}}()
    sizehint!(halton_sequence, N) # Pre-allocate memory for efficiency

    for i = 1:N
        x_raw = halton_1d(i, base_x)
        y_raw = halton_1d(i, base_y)
        push!(halton_sequence, (x_raw, y_raw))
    end

    return halton_sequence
end

function generate_extended_halton_sequence_det(N::Int, n::Int)
    raw_halton_sequence = generate_raw_halton_sequence(N, n)

    # map halton sequence through marginal Beta distributions to 
    # change weighting
    weighted_halton_sequence = []
    for pair in raw_halton_sequence
        weighted_x = quantile(Beta(0.7, 0.7), pair[1])
        weighted_y = quantile(Beta(0.7, 0.7), pair[2])
        if weighted_x == 0
            weighted_x = 1/n
        end
        if weighted_y == 0
            weighted_y = 1/n
        end
        weighted_pair = ( Int(ceil(weighted_x * n)), Int(ceil(weighted_y * n)) )

        push!(weighted_halton_sequence, weighted_pair)
    end

    marginals = [(i, i) for i in 1:n]
    extended_halton_sequence = vcat(weighted_halton_sequence, marginals)

    return extended_halton_sequence
end

function main(; n_sims::Int=1000, ns::Vector{Int}=[25, 50, 100, 200])
    # Ensure results directory exists relative to this file
    results_dir = joinpath(@__DIR__, "results")
    if !isdir(results_dir)
        mkpath(results_dir)
    end

    for n in ns
        halton_det_sampling_mat = zeros(n, n);

        N = round(Int, 10 * n * log(n));
        extended_halton_sequence_det = generate_extended_halton_sequence_det(N, n);

        for i in 1:n_sims
            for pair in extended_halton_sequence_det
                halton_det_sampling_mat[pair[1], pair[2]] += 1
            end
        end

        halton_det_sampling_mat = halton_det_sampling_mat ./ n_sims

        reverse_color_stops = [
            (0.0,  "white"),
            (0.1, "lightblue"),
            (0.5, "royalblue"),
            (1, "darkblue")
            ]
        reverse_steep_cmap = PyPlot.matplotlib.colors.LinearSegmentedColormap.from_list("reverse_steep_cmap", reverse_color_stops)

        figure(1); clf()
        imshow(halton_det_sampling_mat, cmap=reverse_steep_cmap, aspect="equal", 
               vmin=0, vmax=maximum(halton_det_sampling_mat),
               extent=(0.5, n + 0.5, n + 0.7, 0.7))
        title("n = $n", fontsize=20)
        xlabel("Ending index", fontsize=20)
        ylabel("Starting index", fontsize=20) 
        xticks(fontsize=14)
        yticks(fontsize=14)
        tight_layout()
        cb = colorbar()
        cb.set_label("# of inclusions\nin GeneratePairs(n)", fontsize=20)

        # Save to the repository-relative results directory
        outpath = joinpath(results_dir, "halton_det_sampling_mat_n$(n).png")
        savefig(outpath, dpi=200)
        PyPlot.close("all")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

