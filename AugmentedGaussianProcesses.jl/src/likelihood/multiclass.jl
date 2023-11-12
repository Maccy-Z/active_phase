import LinearAlgebra as LA

mutable struct MultiClassLikelihood{L} <: AbstractLikelihood
    invlink::L
    n_class::Int # Number of classes
    class_mapping::Vector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    function MultiClassLikelihood(invlink::L, n_class::Int) where {L}
        return new{L}(invlink, n_class)
    end
    function MultiClassLikelihood(invlink::L, n_class, class_mapping, ind_mapping) where {L}
        return new{L}(invlink, n_class, class_mapping, ind_mapping)
    end
end

function MultiClassLikelihood(invlink::AbstractLink, ylabels::AbstractVector)
    return MultiClassLikelihood(
        invlink,
        length(ylabels),
        ylabels,
        Dict(value => key for (key, value) in enumerate(ylabels)),
    )
end

n_latent(l::MultiClassLikelihood) = n_class(l)

n_class(l::MultiClassLikelihood) = l.n_class

function (l::MultiClassLikelihood)(f::AbstractVector)
    return l.invlink(f)
end

function (l::MultiClassLikelihood)(y::Integer, f::AbstractVector)
    return l.invlink(f)[y]
end

function Base.show(io::IO, l::MultiClassLikelihood)
    return print(io, "Multiclass Likelihood (", n_class(l), " classes, $(l.invlink) )")
end

# Convert one-hot array to 1D vector
function one_hot_to_indices(bit_array_2d::BitArray{2})
    # Map the 'findfirst' function over the rows of the BitArray
    return map(findfirst, eachrow(bit_array_2d))
end

function one_hot_to_indices(y::Any)
    return y
end

## Return the labels in a vector of vectors for multiple outputs ##
function treat_labels!(y::AbstractArray{T,N}, likelihood::MultiClassLikelihood) where {T,N}
    y = one_hot_to_indices(y)
    # N <= 1 || error("Target should be a vector of labels")
    init_multiclass_likelihood!(likelihood, y) # Initialize a mapping if not existing
    return create_one_hot(likelihood, y)
end

function init_multiclass_likelihood!(l::MultiClassLikelihood, y::AbstractVector)
    if !isdefined(l, :ind_mapping)
        create_mapping!(l, y)
    end
end

view_y(::MultiClassLikelihood, y::BitMatrix, i::AbstractVector) = view(y, i, :)

onehot_to_ind(y::AbstractVector) = findfirst(y .== 1)

function Distributions.loglikelihood(
    l::MultiClassLikelihood, y::AbstractVector, f::AbstractVector
)
    return loglikelihood(l, onehot_to_ind(y), f)
end

function create_mapping!(l::MultiClassLikelihood, y::AbstractVector)
    num_latent = n_latent(l)
    if !isdefined(l, :class_mapping)
        l.class_mapping = unique(y)
        if length(l.class_mapping) <= num_latent &&
            issubset(l.class_mapping, collect(1:num_latent))
            l.class_mapping = collect(1:num_latent)
        elseif length(l.class_mapping) > num_latent
            throw(
                ErrorException(
                    "The number of unique labels in the data : $(l.class_mapping) is not of the same size then the predefined class number ; $num_latent",
                ),
            )
        end
    end
    return l.ind_mapping = Dict(value => key for (key, value) in enumerate(l.class_mapping))
end

# Given the labels, return one hot encoding, and the mapping of each class
function create_one_hot(l::MultiClassLikelihood, y)
    issubset(unique(y), l.class_mapping) ||
        error("Some labels of y are not part of the expect labels")
    Y = falses(length(y), n_class(l))
    for i in 1:length(y)
        for j in 1:n_class(l)
            if y[i] == l.class_mapping[j]
                Y[i, j] = true
                break # Once we found the right one, we get out
            end
        end
    end
    return Y
end



# function compute_proba(
#     l::MultiClassLikelihood,
#     μ::Tuple{Vararg{<:AbstractVector{T}}},
#     σ²::Tuple{Vararg{<:AbstractVector{T}}},
#     nSamples::Integer=200,
# ) where {T<:Real}
#     K = n_class(l) # Number of classes
#     n = length(μ[1]) # Number of test points
#     μ = hcat(μ...) # Concatenate means together
#     μ = [μ[i, :] for i in 1:n] # Create one vector per sample
#     σ² = hcat(σ²...) # Concatenate variances together
#     σ² = [σ²[i, :] for i in 1:n] # Create one vector per sample
#     pred = zeros(T, n, K) # Empty container for the predictions
#     for i in 1:n
#         # p = MvNormal(μ[i],sqrt.(abs.(σ²[i])))
#         # p = MvNormal(μ[i],sqrt.(max.(eps(T),σ²[i]))) #WARNING DO NOT USE VARIANCE
#         pred[i, :] .= l(μ[i])
#         # for _ in 1:nSamples
#         # end
#     end
#     return NamedTuple{Tuple(Symbol.(l.class_mapping))}(eachcol(pred))
# end

function compute_proba_mean(
    l::MultiClassLikelihood,
    μ::Tuple{Vararg{<:Vector{T}}},
) where {T<:Real}

    K = n_class(l) # Number of classes
    n = length(μ[1]) # Number of test points
    μ = hcat(μ...) # Concatenate means together
    μ = [μ[i, :] for i in 1:n] # Create one vector per sample
    pred = zeros(T, n, K) # Empty container for the predictions

    for i in 1:n
        pred[i, :] .= l(μ[i])
    end
    return pred #NamedTuple{Tuple(Symbol.(l.class_mapping))}(eachcol(pred))
end


# For full covariance matrix
function compute_proba_full(
    l::MultiClassLikelihood,
    μ::Tuple{Vararg{<:Vector{T}}},
    σ²::Union{  Tuple{
                    Vararg{<:LinearAlgebra.Symmetric{T}}},  # For full covariance matrix
                Tuple{Vararg{<:Vector{T}}}},            # For diagonal covariance matrix
    nSamples::Integer;
    sampler::FastGP
) where {T<:Real}
    K = n_class(l) # Number of classes
    n = length(μ[1]) # Number of test points
    pred = zeros(T, n, K) # Empty container for the predictions

    gaussians = sample_multivariate_normals(sampler, μ, σ², n_repeats=nSamples) # Shape (K, n, nSamples)
    gaussians = cat(gaussians..., dims=3)     # Shape (n, nSamples, K)
    gaussians = reshape(gaussians, :, K)      # Shape (n * nSamples, K)

    probs = mapslices(l, gaussians, dims=2)  # Shape (n, nSamples, K)
    probs = reshape(probs, n, nSamples, K)
    means = mean(probs, dims=2)         # Shape (n, 1, K)
    means = dropdims(means, dims=2)   # Shape (n, K)

    stds = std(probs, dims=2)         # Shape (n, 1, K)
    stds = dropdims(stds, dims=2)   # Shape (n, K)
    return (means, stds)
end

# # For diagonal covariance matrix
# function compute_proba_full(
#     l::MultiClassLikelihood,
#     μ::Tuple{Vararg{<:Vector{T}}},
#     σ²::Tuple{Vararg{<:Vector{T}}},
#     nSamples::Integer;
#     sampler::FastGP
# ) where {T<:Real}
#     K = n_class(l) # Number of classes
#     n = length(μ[1]) # Number of test points
#     pred = zeros(T, n, K) # Empty container for the predictions
#
#
#     gaussians = sample_diagonal_normals(sampler, μ, σ², n_repeats=nSamples) # Shape (K, n, nSamples)
#
#     gaussians = cat(gaussians..., dims=3)     # Shape (n, nSamples, K)
#     gaussians = reshape(gaussians, :, K)      # Shape (n * nSamples, K)
#
#     probs = mapslices(l, gaussians, dims=2)  # Shape (n, nSamples, K)
#     probs = reshape(probs, n, nSamples, K)
#     means = mean(probs, dims=2)         # Shape (n, 1, K)
#     means = dropdims(means, dims=2)   # Shape (n, K)
#
#     stds = std(probs, dims=2)         # Shape (n, 1, K)
#     stds = dropdims(stds, dims=2)   # Shape (n, K)
#     return (means, stds)
# end


function expec_loglike(
    model::AbstractGPModel{T,<:MultiClassLikelihood,<:NumericalVI}
) where {T}
    return compute_log_expectations(model)
end

function Distributions.loglikelihood(l::MultiClassLikelihood, y::AbstractVector, fs)
    return loglikelihood.(l, y, [getindex.(fs, i) for i in 1:length(y)])
end

function ∇loglikehood(l::MultiClassLikelihood, y::AbstractVector, fs)
    return ∇loglikelihood.(l, y, [getindex.(fs, i) for i in 1:length(y)])
end

include("softmax.jl")
include("logisticsoftmax.jl")
