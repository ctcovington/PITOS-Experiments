ENV["MPLBACKEND"] = "Agg" # comment this out in order to see plots in interactive mode
using HypothesisTests
using Distributions
using PyPlot
using SpecialFunctions
using StatsBase
using Random
using Printf
using DelimitedFiles
using CSV
using Tables
using DataFrames

const JULIA_MODULE_PATH = joinpath("PITOS", "src", "PITOS.jl")

# include and use the Julia implementation
include(JULIA_MODULE_PATH)
using .PITOS

include("Halton.jl")

struct PointMass{T} <: DiscreteUnivariateDistribution
    value::T
end


Base.rand(d::PointMass) = d.value
Base.rand(d::PointMass, n::Int) = fill(d.value, n)
Distributions.pdf(d::PointMass, x) = x == d.value ? 1.0 : 0.0
Distributions.mean(d::PointMass) = d.value
Distributions.var(d::PointMass) = 0.0
Distributions.logpdf(d::PointMass, x::Real) = x == d.value ? 0.0 : -Inf
Distributions.support(d::PointMass) = d.value
Distributions.minimum(d::PointMass) = d.value
Distributions.maximum(d::PointMass) = d.value
Distributions.value_support(d::PointMass) = d.value


# ____________________________________________________________________________
# Define tests

ns = [10,20,30,40,50,75,100,125,150,175,200]
nns = length(ns)

# Cramer-von Mises test
CM(x) = (n=length(x); i_n=findfirst(n.==ns); (isnothing(i_n) ? error("No CM null for n=$n.") : mean(CM_nulls[:,i_n] .> CM_test_statistic(x))))
CM_test_statistic(x) = (n=length(x); k=(1:n); 1/(12*n) + sum(((2*k .- 1)/(2*n) - sort(x)).^2))

# Compute null distributions for CM
nnull = 10^5 # TODO: used 10^5 for paper results
CM_nulls = zeros(nnull,nns)
for (i_n,n) in enumerate(ns)
    CM_nulls[:,i_n] = [CM_test_statistic(rand(n)) for i=1:nnull]
end

cauchy_combination(pvalues) = ccdf(Cauchy(), mean(tan.(pi*(0.5 .- pvalues))))

# Kolmogorov-Smirnov
KS(x) = pvalue(HypothesisTests.ExactOneSampleKSTest(x, Uniform(0,1)))

# Anderson-Darling
AD(x) = pvalue(HypothesisTests.OneSampleADTest(x, Uniform(0,1)))

# Neyman-Barton
# (This is N_2 in Blinov and Lemeshko (2014).)
function NB(x)
    n = length(x)
    y = x .- 0.5
    V1 = sqrt(n)*mean(2*sqrt(3)*y)
    V2 = sqrt(n)*mean(sqrt(5)*(6*y.^2 .- 0.5))
    t = V1^2 + V2^2
    p = ccdf(Chisq(2),t)
    return p
end

T_labels = ["PITOS (ours)", "Anderson-Darling", "Neyman-Barton", "Kolmogorov-Smirnov", "Cramér–von Mises"]
Ts = [pitos, AD, NB, KS, CM]


nTs = length(Ts)

# --- Define a color list for the methods ---
method_colors = PyPlot.get_cmap("tab10").colors


# ____________________________________________________________________________
# Define distributions for testing

include("Transformed_Laplace.jl")
include("Transformed_TDist.jl")

distributions = [(Uniform(0,1),"Uniform","Uniform(0,1)",0),
                 (Beta(1.6,1.6),"B_1p6_1p6","Beta(1.6,1.6)",2),
                 (Beta(1.2,0.8),"B_1p2_p8","Beta(1.2,0.8)",1),
                 (Beta(0.6,0.6),"B_p6_p6","Beta(0.6,0.6)",2),
                 (Transformed_Laplace(0,1),"transformed_laplace",L"$\Phi(\mathrm{Laplace}(0,1))$",3),
                 (DiscreteNonParametric((1:99)/100, ones(99)/99),"discrete_uniform",L"$\mathrm{U}(\{ 0.01, 0.02, \ldots, 0.99 \})$",6),
                 (MixtureModel([Uniform(0,1),Uniform(0,0.01),Uniform(0.99,1)],[0.9,0.05,0.05]),"bump_edges","Bump (edges)",5),
                 (Uniform(0.05,0.95),"gap_edges","Gap (edges)",4),
                 (MixtureModel([Uniform(0,1),Uniform(0.2,0.3),Uniform(0.7,0.8)],[0.7,0.15,0.15]),"bump_sides","Bump (sides)",5),
                 (MixtureModel([Uniform(0,0.2),Uniform(0.3,0.7),Uniform(0.8,1)],[2/8,4/8,2/8]),"gap_sides","Gap (sides)",5),
                 (MixtureModel([Uniform(0,1),Uniform(0.45,0.55)],[0.85,0.15]),"bump_middle","Bump (middle)",5),
                 (MixtureModel([Uniform(0,0.45),Uniform(0.55,1)],[0.5,0.5]),"gap_middle","Gap (middle)",5)
                 ]
regular_indices = 1:6
stark_failure_indices = 7:12
                 

dists = [d[1] for d in distributions]  # distribution
tags = [d[2] for d in distributions]  # tag for filenames
labels = [d[3] for d in distributions]  # label
groups = [d[4] for d in distributions]  # group number

nds = length(distributions)


# ____________________________________________________________________________
# Plot densities

xs = 0.001:0.001:0.999

for d = 1:nds
    println("Plotting density for $(labels[d]) ...")
    figure(1); clf(); PyPlot.grid(lw=0.2)
    plot(xs, pdf.(dists[d],xs))
    title("$(labels[d])", fontsize=24)
    xlabel("Value", fontsize=24)
    ylabel("Density", fontsize=24)
    xlim(0,1)
    ylim(0,ylim()[2])
    tick_params(axis="both", which="major", labelsize=12)
    PyPlot.subplots_adjust(top=0.88) 
    tight_layout()
    savefig("results/density-G$(groups[d])-$(tags[d]).png",dpi=200)
    PyPlot.close(1)
end


# ____________________________________________________________________________
# Compute power curves

nreps = 10^5 # TODO: used 10^5 for paper results
alpha = 0.05


# Compute null distributions for LRTs
nnull = 10^5 # TODO: used 10^5 for paper results
lrt_nulls = zeros(nnull,nns,nds)
for (i_n,n) in enumerate(ns)
    for d = 1:nds
        lrt_nulls[:,i_n,d] = [sum(logpdf.(dists[d],rand(n))) for i=1:nnull]
    end
end

# Likelihood ratio test (LRT)
function LRT(x,i_n,d)
    t = sum(logpdf.(dists[d],x))
    return mean(lrt_nulls[:,i_n,d] .>= t)
end

# --- Grid Plot Initializations ---
# Grid for "Stark Failure" Power Curves
fig_grid_failures, axs_grid_failures = PyPlot.subplots(2, 3, figsize=(24, 14), sharex=true, sharey=true)
axs_grid_failures = vec(axs_grid_failures)
grid_plot_idx_failures = 1 

# Grid for "Stark Failure" Density Plots
fig_density_failures, axs_density_failures = PyPlot.subplots(2, 3, figsize=(24, 14), sharex=true, sharey=false)
axs_density_failures = vec(axs_density_failures)
density_grid_plot_idx_failures = 1

# Grid for "Regular" Power Curves
fig_grid_regular, axs_grid_regular = PyPlot.subplots(2, 3, figsize=(24, 14), sharex=true, sharey=true)
axs_grid_regular = vec(axs_grid_regular)
grid_plot_idx_regular = 1

# Grid for "Regular" Density Plots
fig_density_regular, axs_density_regular = PyPlot.subplots(2, 3, figsize=(24, 14), sharex=true, sharey=false)
axs_density_regular = vec(axs_density_regular)
density_grid_plot_idx_regular = 1

# look for csv file with power results, otherwise compute new power results
power_csv = "results/power-G.csv"

# --- Main Loop --
# Use NaN as a placeholder for missing values
const MISSING_VAL = NaN

# Declare power in the outer scope
local power

if isfile(power_csv)
    println("Loading power results from $power_csv ...")
    # Read the data and ensure it's Float64 to accommodate NaNs
    power_data = CSV.read(power_csv, DataFrame, header=false)
    power = reshape(Matrix{Float64}(power_data), (nds, nns, length(Ts)+1))
    println("Power results loaded.")
else
    println("No existing power results found. Initializing new results matrix...")
    # if it doesn't exist, initialize with our missing value marker
    power = fill(MISSING_VAL, (nds, nns, length(Ts)+1))
    power_table = reshape(power, (nds*nns, length(Ts)+1))
    CSV.write(power_csv, Tables.table(power_table), writeheader=false)
    println("New results matrix initialized and saved.")
end

println("Starting/resuming power calculations...")

for d = 1:nds
    # declare that we are modifying the global counter variables
    global grid_plot_idx_failures
    global density_grid_plot_idx_failures
    global grid_plot_idx_regular
    global density_grid_plot_idx_regular # NEW

    for (i_n,n) in enumerate(ns)
        println("Checking $(labels[d]) with n = $n ...")
        
        # flag to track if we do any new computations in this loop
        work_done_in_this_loop = false

        # pre-generate data for this (d, i_n) combination
        x = [rand(dists[d],n) for rep = 1:nreps]
        N = round(Int, 10*n * log(n))

        for (i_T,T) in enumerate(Ts)
            # check if the cell is missing
            if isnan(power[d, i_n, i_T])
                println("    Computing power for T = $(T) ...")
                power[d, i_n, i_T] = mean([(T(x[rep]) .< alpha) for rep = 1:nreps])
                work_done_in_this_loop = true
            else
                println("    Skipping T = $(T), result already exists.")
            end
        end
        
        # check the 'end' case (LRT)
        if isnan(power[d, i_n, end])
            println("    Computing power for LRT ...")
            power[d, i_n, end] = mean([(LRT(x[rep], i_n, d) .< alpha) for rep = 1:nreps])
            work_done_in_this_loop = true
        else
            println("    Skipping LRT, result already exists.")
        end

        # save progress at the end of the i_n loop, if new work was done
        if work_done_in_this_loop
            println("    ... New results computed. Saving progress to $power_csv ...")
            power_table = reshape(power, (nds*nns, length(Ts)+1))
            CSV.write(power_csv, Tables.table(power_table), writeheader=false)
            println("    ... Progress saved.")
        else
            println("    ... No new results for n = $n, skipping save.")
        end

    end
    println("Power computation for $(labels[d]) completed.")
end

println("All power computations finished.")
    
for d = 1:nds
    global grid_plot_idx_failures
    global density_grid_plot_idx_failures
    global grid_plot_idx_regular
    global density_grid_plot_idx_regular
    
    # --- Plot 1: Individual Power Curve (Object-Oriented Style) ---
    fig_power_indiv, ax_power_indiv = PyPlot.subplots() # Create a new, separate figure
    ax_power_indiv.grid(lw=0.2)
    for (i_T,T) in enumerate(Ts)
        ax_power_indiv.plot(ns,vec(power[d,:,i_T]), label=T_labels[i_T], color=method_colors[i_T], lw=3.0)
    end
    ax_power_indiv.plot(ns,vec(power[d,:,end]), label="Likelihood ratio test (Oracle)", color="black", linestyle="--", lw=3.0)
    ax_power_indiv.legend()
    ax_power_indiv.set_title(labels[d], fontsize=36)
    ax_power_indiv.set_xlabel("Sample size", fontsize=36)
    ax_power_indiv.set_ylabel("Power", fontsize=36)
    ax_power_indiv.tick_params(axis="both", which="major", labelsize=18)
    fig_power_indiv.tight_layout()
    fig_power_indiv.savefig("results/power-G$(groups[d])-$(tags[d]).png",dpi=200)
    PyPlot.close(fig_power_indiv) # Close the specific figure object

    # --- Add Power Plot to FAILURES Grid ---
    if d in stark_failure_indices && grid_plot_idx_failures <= length(axs_grid_failures)
        ax = axs_grid_failures[grid_plot_idx_failures]
        for (i_T,T) in enumerate(Ts)
            ax.plot(ns, vec(power[d,:,i_T]), label=T_labels[i_T], color=method_colors[i_T], lw=3.0)
        end
        ax.plot(ns, vec(power[d,:,end]), label="Likelihood ratio test (Oracle)", color="black", linestyle="--", lw=3.0)
        ax.set_title(labels[d], fontsize=28); ax.grid(lw=0.2)
        ax.tick_params(axis="both", which="major", labelsize=16)
        grid_plot_idx_failures += 1
    end

    # --- Add Power Plot to REGULAR Grid ---
    if d in regular_indices && grid_plot_idx_regular <= length(axs_grid_regular)
        ax = axs_grid_regular[grid_plot_idx_regular]
        for (i_T,T) in enumerate(Ts)
            ax.plot(ns, vec(power[d,:,i_T]), label=T_labels[i_T], lw = 3.0, color=method_colors[i_T])
        end
        ax.plot(ns, vec(power[d,:,end]), label="Likelihood ratio test (Oracle)", lw = 3.0, color="black", linestyle="--")
        ax.set_title(labels[d], fontsize=28); ax.grid(lw=0.2)
        ax.tick_params(axis="both", which="major", labelsize=16)
        grid_plot_idx_regular += 1
    end

    # --- Save power table ---
    power_200 = vec(power[d,end,:])
    power_200_table = hcat( [T_labels; "Likelihood ratio test (Oracle)"], power_200)
    writedlm("results/power_n200-G$(groups[d])-$(tags[d]).csv", power_200_table, ',')

    # --- Plot 2: Individual Density Plot (Object-Oriented Style) ---
    xs = 0.001:0.001:0.999
    fig_density_indiv, ax_density_indiv = PyPlot.subplots() # Create another new figure
    ax_density_indiv.hist(rand(dists[d], 200), bins=50, density=true, facecolor="lightblue", label="Histogram (n=200)")
    ax_density_indiv.plot(xs, pdf.(dists[d], xs), color="red", lw=3, label="True density")
    ax_density_indiv.set_title("$(labels[d])", fontsize=36)
    ax_density_indiv.set_xlabel("Value", fontsize=36)
    ax_density_indiv.set_ylabel("Density", fontsize=36)
    ax_density_indiv.grid(true, linestyle="--", color="gray", alpha=0.6)
    ax_density_indiv.set_xlim(0, 1)
    ax_density_indiv.set_ylim(bottom=0)
    ax_density_indiv.tick_params(axis="both", which="major", labelsize=18)
    fig_density_indiv.tight_layout()
    fig_density_indiv.savefig("results/density_plus_n200_histogram-G$(groups[d])-$(tags[d]).png",dpi=200)
    PyPlot.close(fig_density_indiv) # Close the specific figure object

    # --- Add Density Plot to FAILURES Grid ---
    if d in stark_failure_indices && density_grid_plot_idx_failures <= length(axs_density_failures)
        ax_density = axs_density_failures[density_grid_plot_idx_failures]
        if tags[d] == "discrete_uniform"
            ax_density.hist(rand(dists[d], 200), bins=200, density=true, facecolor="lightblue", label="Histogram (n=200)")
        else
            ax_density.hist(rand(dists[d], 200), bins=50, density=true, facecolor="lightblue", label="Histogram (n=200)")
            ax_density.plot(xs, pdf.(dists[d], xs), color="red", lw=3, label="True density")
        end
        ax_density.set_title(labels[d], fontsize=28)
        ax_density.grid(true, linestyle="--", color="gray", alpha=0.6)
        ax_density.tick_params(axis="both", which="major", labelsize=16)
        density_grid_plot_idx_failures += 1
    end

    # --- Add Density Plot to REGULAR Grid ---
    if d in regular_indices && density_grid_plot_idx_regular <= length(axs_density_regular)
        ax_density = axs_density_regular[density_grid_plot_idx_regular]
        ax_density.hist(rand(dists[d], 200), bins=50, density=true, facecolor="lightblue", label="Histogram (n=200)")
        ax_density.plot(xs, pdf.(dists[d], xs), color="red", lw=3, label="True density")
        ax_density.set_title(labels[d], fontsize=28)
        ax_density.grid(true, linestyle="--", color="gray", alpha=0.6)
        ax_density.tick_params(axis="both", which="major", labelsize=16)
        density_grid_plot_idx_regular += 1
    end
end

# --- Finalize and Save Grid Plots ---

# 1. Finalize Failures Power Grid
if grid_plot_idx_failures > 1
    handles, labels_leg = axs_grid_failures[1].get_legend_handles_labels()

    # Calculate number of columns needed to split the legend into 2 rows
    n_items = length(labels_leg)
    n_cols_2_rows = ceil(Int, n_items / 2)

    # Use the new n_cols_2_rows variable in the legend function
    fig_grid_failures.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=n_cols_2_rows, fontsize=32)
    
    fig_grid_failures.supxlabel("Sample size", fontsize=32); fig_grid_failures.supylabel("Power", fontsize=32)
    for ax in axs_grid_failures
        ax.tick_params(axis="both", which="major", labelsize=24)
    end    

    fig_grid_failures.tight_layout(rect=[0.02, 0.02, 1, 0.85])
    fig_grid_failures.savefig("results/power_grid_failures.png", dpi=200)
    PyPlot.close(fig_grid_failures)
    println("Power grid plot for failures saved to results/power_grid_failures.png")
end

# 2. Finalize Failures Density Grid
if density_grid_plot_idx_failures > 1
    handles_density, labels_density = axs_density_failures[1].get_legend_handles_labels()
    # Thicken only the line object in the legend
    for handle in handles_density
        handle.set_linewidth(3.0)
    end
    fig_density_failures.legend(handles_density, labels_density, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=30)
    fig_density_failures.supxlabel("Value", fontsize=32); fig_density_failures.supylabel("Density", fontsize=32)
    for ax in axs_density_failures
        ax.tick_params(axis="both", which="major", labelsize=24)
    end
    fig_density_failures.tight_layout(rect=[0.02, 0.02, 1, 0.9])
    fig_density_failures.savefig("results/density_grid_failures.png", dpi=200)
    PyPlot.close(fig_density_failures)
    println("Density grid plot for failures saved to results/density_grid_failures.png")
end

# 3. Finalize Regular Power Grid
if grid_plot_idx_regular > 1
    handles, labels_leg = axs_grid_regular[1].get_legend_handles_labels()

    # Calculate number of columns needed to split the legend into 2 rows
    n_items = length(labels_leg)
    n_cols_2_rows = ceil(Int, n_items / 2)

    # Use the new n_cols_2_rows variable in the legend function
    fig_grid_regular.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=n_cols_2_rows, fontsize=32)
    
    fig_grid_regular.supxlabel("Sample size", fontsize=32); fig_grid_regular.supylabel("Power", fontsize=32)
    for ax in axs_grid_regular
        ax.tick_params(axis="both", which="major", labelsize=24)
    end

    # Changed top boundary from 0.9 to 0.88 to make room for the 2-row legend
    fig_grid_regular.tight_layout(rect=[0.02, 0.02, 1, 0.85])
    
    fig_grid_regular.savefig("results/power_grid_regular.png", dpi=200)
    PyPlot.close(fig_grid_regular)
    println("Power grid plot for regular cases saved to results/power_grid_regular.png")
end

# 4. Finalize Regular Density Grid
if density_grid_plot_idx_regular > 1
    handles_density, labels_density = axs_density_regular[1].get_legend_handles_labels()
    # Thicken only the line object in the legend
    fig_density_regular.legend(handles_density, labels_density, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=30)
    fig_density_regular.supxlabel("Value", fontsize=32); fig_density_regular.supylabel("Density", fontsize=32)
    for ax in axs_density_regular
        ax.tick_params(axis="both", which="major", labelsize=24)
    end
    
    fig_density_regular.tight_layout(rect=[0.02, 0.02, 1, 0.9])
    fig_density_regular.savefig("results/density_grid_regular.png", dpi=200)
    PyPlot.close(fig_density_regular)
    println("Density grid plot for regular cases saved to results/density_grid_regular.png")
end

# ____________________________________________________________________________
# power over families of distributions


function plot_ranking(prank_probs, avg_rank, avg_power, T_labels, scenario_label, figpath)
    colors = PyPlot.matplotlib.colors
    nTs = length(T_labels)

    # --- create custom colormaps ---
    red_white_cmap = colors.LinearSegmentedColormap.from_list("red_white", ["red", "white"])
    white_blue_cmap = colors.LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])

    # --- plotting ---
    # Increased figure height to give titles and labels more room
    figure(scenario_label, figsize=(14, 10))
    clf()
    ax = gca() # Get current axes

    # 1. plot the main probability matrix
    im1 = imshow(prank_probs, cmap=white_blue_cmap, origin="upper",
                vmin=0, vmax=1,
                extent=[-0.5, nTs-0.5, nTs-0.5, -0.5])

    # 2. plot the 'average rank' column
    im2 = imshow(reshape(avg_rank, nTs, 1), cmap=red_white_cmap, origin="upper",
                vmin=1, vmax=nTs,
                extent=[nTs-0.5, nTs+0.5, nTs-0.5, -0.5])

    # 3. plot the 'average power' column
    im3 = imshow(reshape(avg_power, nTs, 1), cmap=white_blue_cmap, origin="upper",
                vmin=0, vmax=1,
                extent=[nTs+0.5, nTs+1.5, nTs-0.5, -0.5])


    # --- aesthetics ---
    xlim(-0.5, nTs + 1.5)
    ylim(nTs - 0.5, -0.5)

    # annotate each cell within the heatmap
    combined_data = hcat(prank_probs, avg_rank, avg_power)
    for t in 1:size(combined_data, 1)
        for r in 1:size(combined_data, 2)
            value = combined_data[t, r]
            text_color = "black"

            if r <= nTs # Rank probabilities
                if value > 0.6; text_color = "white"; end
                text(r - 1, t - 1, @sprintf("%.2f", value), ha="center", va="center", color=text_color, fontsize=18)
            elseif r == nTs + 1 # Average rank column
                if value < (1 + nTs) / 2; text_color = "white"; end
                text(r - 1, t - 1, @sprintf("%.2f", value), ha="center", va="center", color=text_color, fontsize=18)
            else # Average power column
                if value > 0.6; text_color = "white"; end
                text(r - 1, t - 1, @sprintf("%.2f", value), ha="center", va="center", color=text_color, fontsize=18)
            end
        end
    end

    # add colorbars with larger font sizes
    cbar1 = colorbar(im2, shrink=0.7, ax=ax)
    cbar1.set_label("Average rank", size=30)
    cbar1.ax.tick_params(labelsize=20)

    cbar2 = colorbar(im1, shrink=0.7, ax=ax)
    cbar2.set_label("Probability / Power", size=30)
    cbar2.ax.tick_params(labelsize=20)

    # add title and y-label
    title(scenario_label, fontsize=36)
    ylabel("Test", fontsize=36)

    # set ticks for all columns
    xticks(0:(nTs+1), [string.(1:nTs)..., "Average rank", "Average power"], rotation=45, ha="right", fontsize=24)
    yticks(0:nTs-1, T_labels, fontsize=24)

    ax.text((nTs - 1) / 2, -0.1, "Rank",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=36)

    tight_layout()

    # save the figure, ensuring the bounding box includes all elements
    savefig(figpath, bbox_inches="tight", dpi=200)
    PyPlot.close()
end

function gen_std_t(df,n)
    x = rand(TDist(df),n)
    x = x ./ sqrt(df/(df-2))
    return x
end

function gen_outliers(n, op, outlier_ub)
    x = zeros(n)
    for i in 1:n
        if rand() > op
            x[i] = rand(Uniform(0,1))
        else
            x[i] = rand(Uniform(0, outlier_ub))
        end
    end
    return x
end

function gen_outliers_right(n, op, outlier_ub)
    x = zeros(n)
    for i in 1:n
        if rand() > op
            x[i] = rand(Uniform(0,1))
        else
            x[i] = rand(Uniform(outlier_ub, 1))
        end
    end
    return x
end

random_censoring = false
scenarios = ["nearly_uniform_beta", "symmetric_lighttailed_beta", "asymmetric_lighttailed_beta", "symmetric_heavytailed_beta", "asymmetric_heavytailed_beta", "outliers","random_bump", "random_gap"]
scenario_labels = ["Nearly uniform", "Symmetric light-tailed", "Asymmetric light-tailed", 
                   "Symmetric heavy-tailed", "Asymmetric heavy-tailed", "Outliers", "Random bump", "Random gap"]

alpha = 0.05  # level
n = 100  # number of data points in each data set
nreps = 1000  # number of data sets per distribution TODO: used 1000 for paper results
ndists = 1000 # number of random distributions to simulate TODO: used 1000 for paper results

N = round(Int, 10*n * log(n))

for (scenario, scenario_label) in zip(scenarios, scenario_labels)
    power_file = "results/power_$(scenario).csv"

    if isfile(power_file)
        println("Loading power results from $power_file ...")
        power_data = CSV.read(power_file, DataFrame, header=false)
        power = Matrix(power_data)
        println("Power results loaded.")
    else
        if scenario=="symmetric_lighttailed_beta"
            K = 1
            D_m = PointMass(0.5)
            D_s = Gamma(5, 1/2)
        elseif scenario=="asymmetric_lighttailed_beta"
            K = 1
            D_m = Beta(10,10)
            D_s = Gamma(5, 1/2)
        elseif scenario=="symmetric_heavytailed_beta"
            K = 1
            D_m = PointMass(0.5)
            D_s = Gamma(3, 1/2)
        elseif scenario=="asymmetric_heavytailed_beta"
            K = 1
            D_m = Beta(10,10)
            D_s = Gamma(3, 1/2)
        elseif scenario=="nearly_uniform_beta"
            K = 1
            D_m = Beta(50,50)
            D_s = Gamma(2*50, 1/50)
        elseif scenario=="random_bump"
            K = 1
            D_m = 1
            D_s = 1
            spike_loc_dist = Uniform(0.001,0.999)
            spike_weight_dist = Uniform(0.01,0.1)
        elseif scenario=="random_gap"
            K = 1
            D_m = 1
            D_s = 1
            gap_loc_dist = Uniform(0.1,0.9)
            gap_width_dist = Uniform(0.05, 0.2)
        elseif scenario=="uniform_beta_mixture"
            K = 1
            D_m = 1
            D_s = 1
            Uniform_weight_dist = Uniform(0.5, 0.8)
            Beta_param_dist = Uniform(7,13)
        elseif scenario=="uniform"
            K = 1
            D_m = PointMass(0.5)
            D_s = PointMass(2.0)
        elseif scenario=="t"
            K = 1
            df_dist = Uniform(2, 8)
        elseif scenario=="outliers"
            K = 1
            op_dist = Uniform(0, 0.1)
            outlier_ub_dist = Uniform(0, 0.01)
        elseif scenario=="outliers_right"
            K = 1
            op_dist = Uniform(0, 0.1)
            outlier_ub_dist = Uniform(0.99, 1)
        else
            error("Unknown scenario: $scenario")
        end

        gamma = K  # concentration parameter of Dirichlet prior on mixture weights

        function generate_samples(m,s,w,c,n,random_censoring)
            x = zeros(n)
            i = 1
            while i<=n
                zi = rand(Categorical(w))
                xi = rand(Beta(s[zi]*m[zi], s[zi]*(1-m[zi])))
                if random_censoring && ((c[1] < xi < c[2]) || (c[2] < c[1] < xi) || (xi < c[2] < c[1])); continue; end
                x[i] = xi
                i = i + 1
            end
            return x
        end

        # Compute power for each random distribution
        power = zeros(ndists,nTs)
        ms = zeros(ndists, K)
        ss = zeros(ndists, K)
        ws = zeros(ndists, K)
        dfs = zeros(ndists)
        ops = zeros(ndists) 
        outlier_ubs = zeros(ndists)
        spike_locs = zeros(ndists)
        spike_weights = zeros(ndists)
        gap_locs = zeros(ndists)
        gap_widths = zeros(ndists)
        Uniform_weights = zeros(ndists)
        Beta_params = zeros(ndists)


        for d = 1:ndists
            if mod(d,10)==0; println("$d/$ndists"); end
            if endswith(scenario,"beta")
                m = rand(D_m,K)  # component means
                s = rand(D_s,K)  # component scales
                w = rand(Dirichlet(K,gamma))  # component weights
                
                # for light-tailed beta, accept m,s only if m*s and (1-m)*s are both > 1
                if occursin("lighttailed",scenario)
                    while any(m.*s .<= 1) || any((1 .- m).*s .<= 1)
                        m = rand(D_m,K)
                        s = rand(D_s,K)
                    end
                end

                # for heavy-tailed beta, accept m,s only if m[k]*s[k] <= 1 or (1-m[k])*s[k] <= 1 for all k
                if occursin("heavytailed",scenario)
                    while any((m.*s .> 1) .& ((1 .- m).*s .> 1))
                        m = rand(D_m,K)
                        s = rand(D_s,K)
                    end
                end

                ms[d,:] = m
                ss[d,:] = s
                ws[d,:] = w
            elseif scenario=="mixture_normal"
                m = rand(D_m,K)  # component means
                s = rand(D_s,K)  # component scales
                w = rand(Dirichlet(K,gamma))  # component weights
                ms[d,:] = m
                ss[d,:] = s
                ws[d,:] = w
            elseif scenario=="t"
                df = rand(df_dist)
                dfs[d] = df
            elseif scenario=="outliers"
                op = rand(op_dist)
                ops[d] = op
                outlier_ub = rand(outlier_ub_dist)
                outlier_ubs[d] = outlier_ub
            elseif scenario=="outliers_right"
                op = rand(op_dist)
                ops[d] = op
                outlier_ub = rand(outlier_ub_dist)
                outlier_ubs[d] = outlier_ub
            elseif scenario=="random_bump"
                spike_loc = rand(spike_loc_dist)
                spike_weight = rand(spike_weight_dist)
                spike_locs[d] = spike_loc
                spike_weights[d] = spike_weight
            elseif scenario=="random_gap"
                gap_loc = rand(gap_loc_dist)
                gap_width = rand(gap_width_dist)
                gap_locs[d] = gap_loc
                gap_widths[d] = gap_width
            elseif scenario=="uniform_beta_mixture"
                Uniform_weight = rand(Uniform_weight_dist)
                Beta_param = rand(Beta_param_dist)
                Uniform_weights[d] = Uniform_weight
                Beta_params[d] = Beta_param
            end
            
            c = (c1=rand(); c2=c1+0.1; c2=(c2 > 1 ? c2-1 : c2); [c1,c2])  # random_censoring interval

            if endswith(scenario,"beta")
                x = [[rand(Beta(s[z]*m[z], s[z]*(1-m[z]))) for z in rand(Categorical(w),n)] for rep = 1:nreps]
            elseif scenario=="t"
                x = [cdf(Normal(), gen_std_t(df,n)) for rep = 1:nreps]
            elseif scenario=="mixture_normal"
                x = [[cdf(Normal(), rand(Normal(m[z], s[z]))) for z in rand(Categorical(w),n)] for rep = 1:nreps]
            elseif scenario=="outliers"
                x = [gen_outliers(n, op, outlier_ub) for rep = 1:nreps]
            elseif scenario=="outliers_right"
                x = [gen_outliers_right(n, op, outlier_ub) for rep = 1:nreps]
            elseif scenario=="random_bump"
                x = [rand(MixtureModel([Uniform(0,1), Uniform(spike_loc-0.001,spike_loc+0.001)], 
                                [1-spike_weight,spike_weight]),n) for rep = 1:nreps]
            elseif scenario=="random_gap"
                gap_lb = gap_loc - gap_width/2
                gap_ub = gap_loc + gap_width/2
                normalizing_constant = gap_lb + (1-gap_ub)
                x = [rand( MixtureModel([Uniform(0, gap_lb), Uniform(gap_ub,1)], [gap_lb/normalizing_constant, (1-gap_ub)/normalizing_constant]), n ) for rep = 1:nreps]
            elseif scenario=="uniform_beta_mixture"
                x = [rand(MixtureModel([Uniform(0,1), Beta(Beta_param, Beta_param)], 
                                [Uniform_weight, 1-Uniform_weight]),n) for rep = 1:nreps]
            end

            for (i_T,T) in enumerate(Ts)
                power[d,i_T] = mean([(T(x[rep]) .< alpha) for rep = 1:nreps])
            end   
        end
        # Save power results to CSV
        CSV.write(power_file, Tables.table(power), writeheader=false)
    end

    # plot distributions
    xs = 0.0001:0.0001:0.9999

    # get list of qualitatively different 10 colors to cycle through
    colors = ["#3b7c70", "#ce9642", "#898e9f", "#3b3a3e", "#e67e22", "#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f1c40f"]

    figure(2); clf(); PyPlot.grid(lw=0.2)
    y_max = 20
    K = 1
    gamma = 1
    ms = zeros(10, K)
    ss = zeros(10, K)
    ws = zeros(10, K)
    dfs = zeros(10)
    ops = zeros(10) 
    outlier_ubs = zeros(10)
    spike_locs = zeros(10)
    spike_weights = zeros(10)
    gap_locs = zeros(10)
    gap_widths = zeros(10)
    Uniform_weights = zeros(10)
    Beta_params = zeros(10)
    for d in 1:10
        if scenario=="symmetric_lighttailed_beta"
            K = 1
            gamma = K
            D_m = PointMass(0.5)
            D_s = Gamma(5, 1/2)
        elseif scenario=="asymmetric_lighttailed_beta"
            K = 1
            D_m = Beta(10,10)
            D_s = Gamma(5, 1/2)
        elseif scenario=="symmetric_heavytailed_beta"
            K = 1
            D_m = PointMass(0.5)
            D_s = Gamma(3, 1/2)
        elseif scenario=="asymmetric_heavytailed_beta"
            K = 1
            D_m = Beta(10,10)
            D_s = Gamma(3, 1/2)
        elseif scenario=="nearly_uniform_beta"
            K = 1
            D_m = Beta(50,50)
            D_s = Gamma(2*50, 1/50)
        elseif scenario=="random_bump"
            K = 1
            D_m = 1
            D_s = 1
            spike_loc_dist = Uniform(0.001,0.999)
            spike_weight_dist = Uniform(0.01,0.1)
        elseif scenario=="random_gap"
            K = 1
            D_m = 1
            D_s = 1
            gap_loc_dist = Uniform(0.1,0.9)
            gap_width_dist = Uniform(0.05, 0.2)
        elseif scenario=="uniform_beta_mixture"
            K = 1
            D_m = 1
            D_s = 1
            Uniform_weight_dist = Uniform(0.5, 0.8)
            Beta_param_dist = Uniform(7,13)
        elseif scenario=="uniform"
            K = 1
            D_m = PointMass(0.5)
            D_s = PointMass(2.0)
        elseif scenario=="t"
            K = 1
            df_dist = Uniform(2, 8)
        elseif scenario=="outliers"
            K = 1
            op_dist = Uniform(0, 0.1)
            outlier_ub_dist = Uniform(0, 0.01)
        elseif scenario=="outliers_right"
            K = 1
            op_dist = Uniform(0, 0.1)
            outlier_ub_dist = Uniform(0.99, 1)
        else
            error("Unknown scenario: $scenario")
        end

        # generate parameters for distribution
        if endswith(scenario,"beta")
            m = rand(D_m,K)  # component means
            s = rand(D_s,K)  # component scales
            w = rand(Dirichlet(K,K))  # component weights
            
            # for light-tailed beta, accept m,s only if m*s and (1-m)*s are both > 1
            if occursin("lighttailed",scenario)
                while any(m.*s .<= 1) || any((1 .- m).*s .<= 1)
                    m = rand(D_m,K)
                    s = rand(D_s,K)
                end
            end

            # for heavy-tailed beta, accept m,s only if m[k]*s[k] <= 1 or (1-m[k])*s[k] <= 1 for all k
            if occursin("heavytailed",scenario)
                while any((m.*s .> 1) .& ((1 .- m).*s .> 1))
                    m = rand(D_m,K)
                    s = rand(D_s,K)
                end
            end

            ms[d,:] = m
            ss[d,:] = s
            ws[d,:] = w
        elseif scenario=="mixture_normal"
            m = rand(D_m,K)  # component means
            s = rand(D_s,K)  # component scales
            w = rand(Dirichlet(K,gamma))  # component weights
            ms[d,:] = m
            ss[d,:] = s
            ws[d,:] = w
        elseif scenario=="t"
            df = rand(df_dist)
            dfs[d] = df
        elseif scenario=="outliers"
            op = rand(op_dist)
            ops[d] = op
            outlier_ub = rand(outlier_ub_dist)
            outlier_ubs[d] = outlier_ub
        elseif scenario=="outliers_right"
            op = rand(op_dist)
            ops[d] = op
            outlier_ub = rand(outlier_ub_dist)
            outlier_ubs[d] = outlier_ub
        elseif scenario=="random_bump"
            spike_loc = rand(spike_loc_dist)
            spike_weight = rand(spike_weight_dist)
            spike_locs[d] = spike_loc
            spike_weights[d] = spike_weight
        elseif scenario=="random_gap"
            gap_loc = rand(gap_loc_dist)
            gap_width = rand(gap_width_dist)
            gap_locs[d] = gap_loc
            gap_widths[d] = gap_width
        elseif scenario=="uniform_beta_mixture"
            Uniform_weight = rand(Uniform_weight_dist)
            Beta_param = rand(Beta_param_dist)
            Uniform_weights[d] = Uniform_weight
            Beta_params[d] = Beta_param
        end

        # set distribution
        if endswith(scenario,"beta")
            dist = Beta(ms[d]*ss[d], ss[d]*(1-ms[d]))
        elseif scenario=="t"
            dist = TDist(dfs[d])
        elseif scenario=="mixture_normal"
            dist = MixtureModel([Normal(ms[d,k], ss[d,k]) for k in 1:K], ws[d,:])
        elseif scenario=="outliers"
            dist = MixtureModel([Uniform(0,1), Uniform(0, outlier_ubs[d])], [1-ops[d], ops[d]])
        elseif scenario=="outliers_right"
            dist = MixtureModel([Uniform(0,1), Uniform(outlier_ubs[d],1)], [1-ops[d], ops[d]])
        elseif scenario=="random_bump"
            dist = MixtureModel([Uniform(0,1), Uniform(spike_locs[d]-0.001,spike_locs[d]+0.001)], [1-spike_weights[d],spike_weights[d]])
        elseif scenario=="random_gap"
            gap_lb = gap_locs[d] - gap_widths[d]/2
            gap_ub = gap_locs[d] + gap_widths[d]/2
            normalizing_constant = 1-gap_widths[d]
            dist = MixtureModel([Uniform(0, gap_lb), Uniform(gap_ub,1)], [gap_lb/normalizing_constant, (1-gap_ub)/normalizing_constant])
        elseif scenario=="uniform_beta_mixture"
            dist = MixtureModel([Uniform(0,1), Beta(Beta_params[d], Beta_params[d])], [Uniform_weights[d], 1-Uniform_weights[d]])
        end
        ys = pdf.(dist, xs)
        
        plot(xs, ys, color=colors[d%10+1], alpha=1)
        y_max = max(y_max, maximum(ys))
    end

    title("$scenario_label", fontsize=30)
    xlabel("Value", fontsize=30)
    ylabel("Density", fontsize=30)
    xticks(fontsize=20)
    yticks(fontsize=20)
    if y_max > 20
        ylim(0, 20)
    end
    tight_layout()
    savefig("results/distributions-$scenario.png", dpi = 200)
    PyPlot.close()

    # Compute ranks
    ranks = zeros(ndists,nTs)
    for d = 1:ndists
        ranks[d,:] = invperm(sortperm(power[d,:]; rev=true))
    end

    # Compute rank probabilities
    prank = zeros(nTs,nTs)
    for t = 1:nTs
        for r = 1:nTs
            prank[t,r] = mean(ranks[:,t].==r)
        end
    end

    # --- Calculate summary statistics for plotting ---
    avg_power = vec(mean(power, dims=1))
    avg_rank = vec(mean(ranks, dims=1))

    display([T_labels avg_power])

    figure(1); clf(); PyPlot.grid(lw=0.2)
    hist(power; density=true, label=T_labels)
    title("Power histograms ($scenario)")
    legend()


    display([T_labels avg_rank])

    display([[""; T_labels] [permutedims(1:nTs); prank]])

    # --- Call the updated plotting function with all the necessary data ---
    plot_ranking(prank, avg_rank, avg_power, T_labels, scenario_label, "results/ranking-$(scenario).png")
    println("Plot saved for scenario: $scenario")
end