using Distributions
using Random
using PyPlot

const JULIA_MODULE_PATH = joinpath("PITOS", "src", "PITOS.jl")

# include and use the Julia implementation
include(JULIA_MODULE_PATH)
using .PITOS

include("Halton.jl")
# set logging level to only show warnings and above
using Logging
Logging.disable_logging(LogLevel(Logging.Info))  # Only show warnings and above

colors = ["#3b7c70", "#ce9642", "#898e9f", "#3b3a3e"]

function plot_ecdfs(no_correction_ps, ps, label, fn, results_dir)
    figure(figsize=(8, 4.5))
    clf();

    # First CDF: No correction
    subplot(1, 2, 1)
    sorted_no_corr = sort(no_correction_ps)
    n_sims = length(sorted_no_corr)
    plot(sorted_no_corr, range(1/n_sims, 1, length=n_sims), color=colors[1], linewidth=3.0)
    plot([0, 1], [0, 1], "k--", linewidth=3.0)
    xlim(0,1)
    ylim(0,1)
    xlabel(raw"$p$", fontsize = 14)
    ylabel("CDF", fontsize = 14)
    xticks(fontsize=10)
    yticks(fontsize=10)
    title("$(label)\nwithout correction", fontsize = 14)
    gca().set_aspect("equal")

    # Second CDF: With correction
    subplot(1, 2, 2)
    sorted_corr = sort(ps)
    plot(sorted_corr, range(1/n_sims, 1, length=n_sims), color=colors[2], linewidth=3.0)
    plot([0, 1], [0, 1], "k--", linewidth=3.0)
    xlim(0,1)
    ylim(0,1)
    xlabel(raw"$p^*$", fontsize = 14)
    ylabel("CDF", fontsize = 14)
    xticks(fontsize=10)
    yticks(fontsize=10)
    title("$(label)\nwith correction", fontsize = 14)
    gca().set_aspect("equal")

    tight_layout()
    savefig(joinpath(results_dir, "$(fn)_ecdfs.png"), dpi=200)
end

function plot_truncated_ecdfs(no_correction_ps, ps, label, fn, t, results_dir)
    figure(figsize=(8, 4.5)) 
    clf();

    # First CDF: No correction
    subplot(1, 2, 1)
    sorted_no_corr = sort(no_correction_ps)
    n_sims = length(sorted_no_corr)
    plot(sorted_no_corr, range(1/n_sims, 1, length=n_sims), color=colors[1], linewidth=3.0)
    plot([0, t], [0, t], "k--", linewidth=3.0)
    xlim(0, t)
    ylim(0, max(t, findfirst((sorted_no_corr.>=0.1))/n_sims))
    xlabel(raw"$p$", fontsize = 14)
    ylabel("CDF", fontsize = 14)
    xticks(0:0.02:0.1, fontsize=10)
    yticks(0:0.02:0.1, fontsize=10)
    title("$(label)\nwithout correction", fontsize = 14)
    gca().set_aspect("equal")

    # Second CDF: With correction
    subplot(1, 2, 2)
    sorted_corr = sort(ps)
    plot(sorted_corr, range(1/n_sims, 1, length=n_sims), color=colors[2], linewidth=3.0)
    plot([0, t], [0, t], "k--", linewidth=3.0)
    xlim(0, t)
    ylim(0, max(t, findfirst((sorted_corr.>=0.1))/n_sims))
    xlabel(raw"$p^*$", fontsize = 14)
    ylabel("CDF", fontsize = 14)
    xticks(0:0.02:0.1, fontsize=10)
    yticks(0:0.02:0.1, fontsize=10)
    title("$(label)\nwith correction", fontsize = 14)
    gca().set_aspect("equal")

    tight_layout()
    savefig(joinpath(results_dir, "$(fn)_trunc_ecdfs.png"), dpi=200)
end

function main(; n::Int=30, n_sims::Int=10^5)
    results_dir = joinpath(@__DIR__, "results")
    if !isdir(results_dir)
        mkpath(results_dir)
    end

    methods = [pitos]
    method_labels = ["PITOS"]
    fig_names = ["PITOS"]

    for (method, label, fn) in zip(methods, method_labels, fig_names)
        no_correction_ps = zeros(n_sims);
        ps = zeros(n_sims);

        for i in 1:n_sims
            # sample from Unif(0,1)
            x = rand(Beta(1,1), n)

            # get p-values with/without correction
            p = method(x)
            no_correction_ps[i] = p 
            ps[i] = min(1, 1.15*p)
        end
        
        plot_ecdfs(no_correction_ps, ps, label, fn, results_dir)
        plot_truncated_ecdfs(no_correction_ps, ps, label, fn, 0.1, results_dir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 