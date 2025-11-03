using Test
using Random
using Distributions
using StatsBase

# Include the module file to be tested
# Assumes the PITOS.jl file is in a sibling 'src' directory
include("../src/PITOS.jl")

# Make the module and its non-exported functions available for testing
using .PITOS
import .PITOS: halton_1d, cauchy_combination, indexed_PITCOS_bidirectional, validate_pairs

# --- Test Suite ---

@testset "PITOS.jl: All Tests" begin

    # Set a seed for reproducibility of all stochastic tests
    Random.seed!(1)

    # ---
    # Section 1: Helper Function Unit Tests
    # ---
    @testset "Helper Functions" begin
        # Test the 1D Halton sequence generator
        @testset "halton_1d" begin
            @test halton_1d(1, 2) == 0.5
            @test halton_1d(2, 2) == 0.25
            @test halton_1d(3, 2) == 0.75
            @test halton_1d(1, 3) == 1/3
            @test halton_1d(2, 3) ≈ 2/3
        end
        
        # Test the core bidirectional PIT calculation
        @testset "indexed_PITCOS_bidirectional" begin
            n = 10
            # A perfectly sorted vector of order statistics from U(0,1)
            xo = collect(1:n) ./ (n + 1.0) 
            
            # For a perfectly uniform sample, the one-sided p-values (u) should be near 0.5
            # Test a pair where start < finish
            u_forward = PITOS.indexed_PITCOS_bidirectional(xo, n, (2, 5))
            @test 0.9 < u_forward <= 1.0 # two-sided p-value should be large
            
            # Test a pair where start > finish
            u_backward = PITOS.indexed_PITCOS_bidirectional(xo, n, (8, 3))
            @test 0.9 < u_backward <= 1.0
        end
    end

    # ---
    # Section 2: Input Validation and Error Handling
    # ---
    @testset "Input Validation" begin
        n = 10
        # Test that invalid pairs throw an ArgumentError
        @test_throws ArgumentError validate_pairs([(0, 5)], n)   # i == 0
        @test_throws ArgumentError validate_pairs([(3, 0)], n)   # j == 0
        @test_throws ArgumentError validate_pairs([(-1, 7)], n)  # i < 0
        @test_throws ArgumentError validate_pairs([(7, 11)], n)  # j > n

        # Test error handling in the main function
        x = rand(n)
        invalid_pairs = [(5, 11)] # Invalid pair for vector of length 10
        @test_throws ArgumentError pitos(x; pairs_sequence=invalid_pairs)
    end

    # ---
    # Section 3: Main `pitos` Functionality Tests
    # ---
    @testset "Main `pitos` Function" begin
        n = 20
        x_rand = rand(n)
        
        # test default mode
        p_default = pitos(x_rand)
        @test 0.0 <= p_default <= 1.0
        
        # test mode with custom pairs
        custom_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        p_custom = pitos(x_rand; pairs_sequence=custom_pairs)
        @test 0.0 <= p_custom <= 1.0
        # Running it again should yield the exact same result
        @test p_custom == pitos(x_rand; pairs_sequence=custom_pairs)

        # Test with an edge-case vector (all identical values)
        # This is extremely unlikely under U(0,1), so p-value should be ~0
        x_identical = fill(0.5, n)
        @test pitos(x_identical) < 1e-5
    end

end