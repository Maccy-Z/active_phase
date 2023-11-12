# function ELBO(model::GP, X, y, pr_mean, kernel)
#         @info "Never Here"
#
#     setpr_mean!(model.f, pr_mean)
#     setkernel!(model.f, kernel)
#     state = compute_kernel_matrices(model, (;), X, true)
#
#     return objective(model, state, y)
# end

## ELBO Section ##
function expec_loglikelihood(
    y, model, state, zero)
    tot = -length(y) * logtwo
    tot += -sum(sum(state.γ .+ eachcol(y))) * logtwo

    total_sum = zero # This is a floating-point number, assuming the operations below yield floats.
    # Iterate over each set of corresponding elements from the input collections.
    for (θ, γ, col_y, gp) in zip(state.θ, state.γ, eachcol(y), model.f)
        current_μ, current_Σ = gp.post.μ, diag(gp.post.Σ)
        # Apply the function body to the current set of elements and add the result to the cumulative sum.
        total_sum += dot(current_μ, (col_y - γ)) - dot(θ, abs2.(current_μ)) - dot(θ, current_Σ)
    end
    tot += total_sum / 2

#     tot += sum(zip(state.θ, state.γ, eachcol(y), μ, Σ)) do (θ, γ, y, μ, Σ)
#         dot(μ, (y - γ)) - dot(θ, abs2.(μ)) - dot(θ, Σ)
#     end / 2

    return tot
end

function ELBO(model::TGP, X, y, pr_means, kernels, state) where {T,L,TGP<:AbstractGPModel{T,L,<:AnalyticVI}}
    #setpr_means!(model, pr_means)
    for (gp, mu) in zip(model.f, pr_means)
        gp.prior.μ₀ = mu
    end

    #setkernels!(model, kernels)
    for (gp, kern) in zip(model.f, kernels)
        gp.prior.kernel = kern
    end

    kernel_matrices = compute_kernel_matrices(kernels, state, X, 0, true)
    model.inference.HyperParametersUpdated = false

    # ELBO function
    tot = zero(T)
    tot += model.inference.ρ* expec_loglikelihood(
            y,
            model,
            state.local_vars,
            zero(T)
        )
    #c = GaussianKL(model, state)
    X_data = (model.data.X for _ in 1:length(model.f))
    c = zero(T)
    for (gp, X, k_mat) in zip(model.f, X_data, kernel_matrices)
        μ, μ₀, Σ, K = gp.post.μ, gp.prior.μ₀(X), gp.post.Σ, k_mat

        mapped_item = (logdet(K) - logdet(Σ) + tr(K \ Σ) + invquad(K, μ - μ₀) - length(μ)) / 2  # Apply the mapping function
        c += mapped_item
    end
    tot -= c
    tot -= ChainRulesCore.ignore_derivatives() do
        ρ(inference(model)) * AugmentedKL(likelihood(model), state.local_vars, y)
    end
    #tot -= extraKL(model, state)

    return tot
end

# function ELBO(model::AbstractGPModel, X, y, pr_means, kernels, Zs, state)
#     @info "Never Here"
#
#     setpr_means!(model, pr_means)
#     setkernels!(model, kernels)
#     setZs!(model, Zs)
#     state = compute_kernel_matrices(model, state, X, true)
#     return objective(model, state, y)
# end

# External ELBO call on internal and new data
# @traitfn function ELBO(model::TGP) where {TGP <: AbstractGPModel; IsFull{TGP}}
#     @info "ELBO3"
#
#     return ELBO(model, input(model.data), output(model.data))
# end
#
# function ELBO(model::AbstractGPModel, X::AbstractMatrix, y::AbstractArray; obsdim=1)
#     @info "Never Here"
#     return ELBO(model, KernelFunctions.vec_of_vecs(X; obsdim), y)
# end

# function ELBO(model::AbstractGPModel, X::AbstractVector, y::AbstractArray)
#     @info "ELBO1"
#
#     y = treat_labels!(y, likelihood(model))
#     state = compute_kernel_matrices(model, (;), X, true)
#     if inference(model) isa AnalyticVI
#         local_vars = init_local_vars(likelihood(model), length(X))
#         local_vars = local_updates!(
#             local_vars,
#             likelihood(model),
#             y,
#             mean_f(model, state.kernel_matrices),
#             var_f(model, state.kernel_matrices),
#         )
#         state = merge(state, (; local_vars))
#     end
#     return ELBO(model, state, y)
# end

# function ELBO(
#     model::OnlineSVGP, state::NamedTuple, X::AbstractMatrix, y::AbstractArray; obsdim=1
# )
# @info "Never here2"
#
#     return ELBO(model, state, KernelFunctions.vec_of_vecs(X; obsdim), y)
# end
#
# function ELBO(model::OnlineSVGP, state::NamedTuple, X::AbstractVector, y::AbstractArray)
#         @info "Never here3"
#
#     y = treat_labels!(y, likelihood(model))
#     state = compute_kernel_matrices(model, state, X, true)
#     if inference(model) isa AnalyticVI
#         local_vars = init_local_vars(likelihood(model), length(X))
#         local_vars = local_updates!(
#             local_vars,
#             likelihood(model),
#             y,
#             mean_f(model, state.kernel_matrices),
#             var_f(model, state.kernel_matrices),
#         )
#         state = merge(state, (; local_vars))
#     end
#     return ELBO(model, state, y)
# end
