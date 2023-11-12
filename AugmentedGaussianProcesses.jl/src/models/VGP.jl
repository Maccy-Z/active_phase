"""
    VGP(args...; kwargs...)

Variational Gaussian Process

## Arguments
- `X::AbstractArray` : Input features, if `X` is a matrix the choice of colwise/rowwise is given by the `obsdim` keyword
- `y::AbstractVector` : Output labels
- `kernel::Kernel` : Covariance function, can be any kernel from KernelFunctions.jl
- `likelihood` : Likelihood of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, see the [`Compatibility Table`](@ref compat_table))

## Keyword arguments
- `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `obsdim::Int=1` : Dimension of the data. 1 : X ∈ DxN, 2: X ∈ NxD
"""
mutable struct VGP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
    TData<:AbstractDataContainer,
    N,
} <: AbstractGPModel{T,TLikelihood,TInference,N}
    data::TData # Data container
    f::NTuple{N,VarLatent{T}} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int #Level of printing information
    atfrequency::Int
    trained::Bool
    length_bounds::Tuple{Real,Real}
    var_bounds::Tuple{Real,Real}
    final_Ks::Union{Nothing,Tuple{Vararg{LinearAlgebra.Cholesky{Float64, Matrix{Float64}}}}}
end

function VGP(
    X::AbstractArray,
    y::AbstractVector,
    kernel::Kernel,
    likelihood::AbstractLikelihood,
    inference::AbstractInference;
    verbose::Int=0,
    optimiser=ADAM(0.01),
    atfrequency::Integer=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    obsdim::Int=1,
    length_bounds::Tuple{Real,Real}=(0.1,5.),
    var_bounds::Tuple{Real,Real}=(0.1,100.),
)
    X, T = wrap_X(X, obsdim)
    y = check_data!(y, likelihood)

    inference isa VariationalInference || error(
        "The inference object should be of type `VariationalInference` : either `AnalyticVI` or `NumericalVI`",
    )
    !isa(likelihood, GaussianLikelihood) || error(
        "For a Gaussian Likelihood you should directly use the `GP` model or the `SVGP` model for large datasets",
    )
    implemented(likelihood, inference) ||
        error("The $likelihood is not compatible or implemented with the $inference")

    data = wrap_data(X, y)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    n_feature = n_sample(data)

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf = ntuple(n_latent(likelihood)) do _
        VarLatent(T, n_feature, kernel, mean, optimiser)
    end

    return VGP{T,typeof(likelihood),typeof(inference),typeof(data),n_latent(likelihood)}(
        data, latentf, likelihood, inference, verbose, atfrequency, false, length_bounds, var_bounds, nothing
    )
end

function Base.show(io::IO, model::VGP)
    return print(
        io,
        "Variational Gaussian Process with a $(likelihood(model)) infered by $(inference(model)) ",
    )
end

function Zviews(m::VGP)
     a = (m.data.X,)
     return a
end
#objective(m::VGP, state, y) = ELBO(m, state, y)

# function objective(m::VGP, state, y)
#     @info "At objective now for VGP"
#     return ELBO(m, state, y)
# end

# function objective(model::VGP, state, y) # where {T,L,TGP<:AbstractGPModel{T,L,<:AnalyticVI};!IsMultiOutput{TGP}}
#
#     tot = 0.#zero(T)
# #     tot +=
# #         ρ(inference(model)) * expec_loglikelihood(
# #             likelihood(model),
# #             inference(model),
# #             y,
# #             mean_f(model, state.kernel_matrices),
# #             var_f(model, state.kernel_matrices),
# #             state.local_vars,
# #         )
#     #c = GaussianKL(model, state)
#     c = 0.
#     for (gp, X, k_mat) in zip(model.f, (model.data.X,model.data.X,model.data.X), state.kernel_matrices)
#         μ, μ₀, Σ, K = gp.post.μ, gp.prior.μ₀(X), gp.post.Σ, k_mat.K
#
#         mapped_item = (logdet(K) - logdet(Σ) + tr(K \ Σ) + invquad(K, μ - μ₀) - length(μ)) / 2  # Apply the mapping function
#         c = c + mapped_item  # Combine with the current result using the reduction function
#     end
#     tot -= c
# #     tot -= ChainRulesCore.ignore_derivatives() do
# #         ρ(inference(model)) * AugmentedKL(likelihood(model), state.local_vars, y)
# #     end
# #     tot -= extraKL(model, state)
#     @info "Total ELBO"
#     println(tot)
#     println(" ")
#     return tot
# end

@traitimpl IsFull{VGP}
