module PITOS

    using Distributions # Required for Beta and Cauchy distributions (cdf, ccdf)
    using StatsBase     # Required for `pweights`
    using Random        # Required for `sample`

    # export public functions
    export pitos

    # --- minimal dependencies ---

    """
        halton_1d(i::Int, b::Int)

    Calculates the i-th element of a 1D Halton sequence with a given base `b`.
    """
    function halton_1d(i::Int, b::Int)
        result = 0.0
        f = 1.0
        
        while i > 0
            f /= b
            result += f * (i % b)
            i = div(i, b) # Integer division
        end
        
        return result
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
    
    function generate_weighted_halton_sequence(N::Int, n::Int)
        raw_halton_sequence = generate_raw_halton_sequence(N, n)
    
        # map halton sequence through marginal Beta distributions to 
        # change weighting
        weighted_halton_sequence = Vector{Tuple{Int, Int}}()
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

    """
        indexed_PITCOS_bidirectional(xo, n, pair)

    Calculates the PITCOS p-value for a given pair of indices (start, finish) 
    from the sorted vector xo.
    """
    function indexed_PITCOS_bidirectional(xo, n, pair)
        (start, finish) = pair
        u = 0.0;

        if start == finish
            u = cdf( Beta(finish, n - finish + 1), xo[finish] )
        elseif start < finish
            if xo[start] == xo[finish]
                u = 0.0
            else
                u = cdf( Beta(finish - start, n - finish + 1), 
                            (xo[finish] - xo[start])/(1.0-xo[start]) 
                            )
            end
        elseif start > finish
            if xo[start] == xo[finish]
                u = 0.0
            else
                u = cdf( Beta(finish, start - finish), 
                            xo[finish] / xo[start] 
                    )
            end
        end

        return 2.0 * min(u, 1.0 - u)
    end

    """
        cauchy_combination(pvalues)

    Performs the Cauchy combination test on a vector of p-values.
    """
    function cauchy_combination(pvalues)
        if any(p -> !(0.0 <= p <= 1.0), pvalues)
            throw(DomainError(pvalues, "All p-values must be in the range [0, 1]."))
        end
        statistic = mean(tan.(pi .* (0.5 .- pvalues)))
        return ccdf(Cauchy(), statistic)
    end

    """
        validate_pairs(pairs::Vector{Tuple{Int, Int}}, n::Int)

    Checks if all pairs are valid for a sorted vector of length n,
    where valid pairs (i, j) satisfy:
    1 <= i <= n, 1 <= j <= n, and i != j. Throws an ArgumentError if violated.
    """
    function validate_pairs(pairs::Vector{Tuple{Int, Int}}, n::Int)
        for (i, j) in pairs
            if i < 1 || j < 1
                throw(ArgumentError("Invalid pair ($i, $j): Indices must be >= 1."))
            end
            if i > n || j > n
                throw(ArgumentError("Invalid pair ($i, $j): Indices must be <= n ($n)."))
            end
        end
        return nothing
    end

    # --- full PITOS implementation ---

    """
        pitos(x::Vector{Float64}; pairs_sequence::Union{Vector{Tuple{Int, Int}}, Nothing}=nothing)

    Performs the PITOS test.
    
    # Arguments
    - `x::Vector{Float64}`: The vector of values to test (will be sorted internally).
    - `pairs_sequence::Union{Vector{Tuple{Int, Int}}, Nothing}`: The sequence of pairs 
        to use. If `nothing`, uses a weighted Halton sequence.


    # Returns
    - `Float64`: The combined p-value (using Cauchy combination).
    """
    function pitos(x::Vector{Float64}; 
                pairs_sequence::Union{Vector{Tuple{Int, Int}}, Nothing}=nothing)

        n = length(x)

        if isnothing(pairs_sequence)
            pairs_sequence = generate_weighted_halton_sequence(round(Int, 10 * n * log(n)), n)
        end
        N = length(pairs_sequence)

        # validate sequences and weights
        validate_pairs(pairs_sequence, n)

        # pre-sort the input vector
        xo = sort(x)
        ps = Vector{Float64}(undef, N)

        for i in 1:N
            pair = pairs_sequence[i]
            ps[i] = indexed_PITCOS_bidirectional(xo, n, pair)
        end
        
        # final combination
        p = cauchy_combination(ps)
        
        return p
    end

end
