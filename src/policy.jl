abstract type AbstractNNPolicy  <: Policy end

struct NNPolicy{P <: Union{MDP, POMDP}, Q, A} <: AbstractNNPolicy 
    problem::P
    network::Q
    action_map::Vector{A}
    n_input_dims::Int64
end

function reset!(policy::NNPolicy)
    Flux.reset!(policy.network)
end

function _action(policy::NNPolicy{P,Q,A}, o::AbstractArray{T, N}) where {P<:Union{MDP,POMDP},Q,A,T<:Real,N}
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        vals = policy.network(obatch)
        return policy.action_map[argmax(vals)] # need to change for continuous
    else 
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

function POMDPPolicies.actionvalues(policy::NNPolicy{P,Q,A}, o::AbstractArray{T,N}) where {P<:Union{MDP,POMDP},Q,A,T<:Real,N}
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        return policy.network(obatch)
    else 
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

function _value(policy::NNPolicy{P}, o::AbstractArray{T,N}) where {P<:Union{MDP,POMDP},T<:Real,N}
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        return maximum(policy.network(obatch))
    else 
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

function POMDPs.action(policy::NNPolicy{P}, s) where {P <: MDP}
    _action(policy, convert_s(Array{Float64}, s, policy.problem))
end

function POMDPs.action(policy::NNPolicy{P}, o) where {P <: POMDP}
    _action(policy, convert_o(Array{Float64}, o, policy.problem))
end

function POMDPs.value(policy::NNPolicy{P}, s) where {P <: MDP}
    _value(policy, convert_s(Array{Float64}, s, policy.problem))
end

function POMDPs.value(policy::NNPolicy{P}, o) where {P <: POMDP}
    _value(policy, convert_o(Array{Float64}, o, policy.problem))
end