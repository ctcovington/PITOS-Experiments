# NOTE: THIS FILE WAS GENERATED USING CO-PILOT

using Distributions

struct Transformed_TDist <: ContinuousUnivariateDistribution
    dof::Float64
end

# Sampling: Y = Phi(X), where X ~ TDist(dof)
function Base.rand(rng::AbstractRNG, d::Transformed_TDist)
    x = rand(rng, TDist(d.dof))
    return cdf(Normal(),x)
end

# Provide sampler method
Distributions.sampler(d::Transformed_TDist) = d

# PDF: f_Y(y) = f_X(x) / phi(x), where x = Phi⁻¹(y)
function Distributions.pdf(d::Transformed_TDist, y::Float64)
    x = quantile(Normal(),y)
    f_x = pdf(TDist(d.dof), x)
    phi = pdf(Normal(), x)
    return f_x / phi
end

# LogPDF
function Distributions.logpdf(d::Transformed_TDist, y::Float64)
    x = quantile(Normal(),y)
    log_f_x = logpdf(TDist(d.dof), x)
    log_phi = logpdf(Normal(), x)
    return log_f_x - log_phi
end

# CDF
function Distributions.cdf(d::Transformed_TDist, y::Float64)
    x = quantile(Normal(),y)
    return cdf(TDist(d.dof), x)
end

# Quantile
function Distributions.quantile(d::Transformed_TDist, q::Float64)
    x = quantile(TDist(d.dof), q)
    return cdf(Normal(),x)
end

# Support
Distributions.minimum(d::Transformed_TDist) = 0.0
Distributions.maximum(d::Transformed_TDist) = 1.0
Distributions.insupport(d::Transformed_TDist, y::Float64) = (0.0 < y < 1.0)


