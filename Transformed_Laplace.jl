# NOTE: THIS FILE WAS GENERATED USING CO-PILOT

using Distributions

struct Transformed_Laplace <: ContinuousUnivariateDistribution
    mu::Float64
    b::Float64
end

# Sampling: Y = Phi(X), where X ~ Laplace(mu, b)
function Base.rand(rng::AbstractRNG, d::Transformed_Laplace)
    x = rand(rng, Laplace(d.mu, d.b))
    return cdf(Normal(),x)
end

# Provide sampler method
Distributions.sampler(d::Transformed_Laplace) = d

# PDF: f_Y(y) = f_X(x) / phi(x), where x = Phi⁻¹(y)
function Distributions.pdf(d::Transformed_Laplace, y::Float64)
    x = quantile(Normal(),y)
    lap = pdf(Laplace(d.mu, d.b), x)
    phi = pdf(Normal(), x)
    return lap / phi
end

# LogPDF: log(f_Y(y)) = log(f_X(x)) - log(phi(x))
function Distributions.logpdf(d::Transformed_Laplace, y::Float64)
    x = quantile(Normal(),y)
    log_lap = logpdf(Laplace(d.mu, d.b), x)
    log_phi = logpdf(Normal(), x)
    return log_lap - log_phi
end

# CDF: P(Y ≤ y) = P(X ≤ Phi⁻¹(y)) = LaplaceCDF(Phi⁻¹(y))
function Distributions.cdf(d::Transformed_Laplace, y::Float64)
    x = quantile(Normal(),y)
    return cdf(Laplace(d.mu, d.b), x)
end

# Quantile: y = Phi(x), so x = Laplace quantile, then y = Phi(x)
function Distributions.quantile(d::Transformed_Laplace, q::Float64)
    x = quantile(Laplace(d.mu, d.b), q)
    return cdf(Normal(),x)
end

# Support
Distributions.minimum(d::Transformed_Laplace) = 0.0
Distributions.maximum(d::Transformed_Laplace) = 1.0
Distributions.insupport(d::Transformed_Laplace, y::Float64) = (0.0 < y < 1.0)


