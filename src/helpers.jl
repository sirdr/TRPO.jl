"""
    get_flat_params_from(model) # flux network
flattens all params of a given neural network and returns a 
concatenated version of them 
"""
function get_flat_params_from(model)
    flat_params = Float64[]
    for param in params(model)
        append!(flat_params, reshape(param.data, length(param)))
    end
    return flat_params 
end

"""
    set_flat_params_to(model, flat_params) # flux network
takes array representing new flattened parameters of network and uses
them to replace the current parameters of the network
"""
function set_flat_params_to(model, flat_params)
    prev_ind = 1
    new_params = Array[]
    for param in params(model)
        flat_size = length(param)
        p = flat_params[prev_ind:prev_ind+(flat_size-1)]
        p = reshape(p, size(param))
        push!(new_params, p)
        prev_ind += flat_size
    end
    Flux.loadparams!(model, new_params)
end


"""
    get_flat_grads_from(model) # flux network
gets all gradients of a given neural network and returns a
flattened version of them
"""

function get_flat_grads_from(model, grad_grad=False)
    flat_grads = Array[]
    for param in params(model)
        if grad_grad
            gg = Tracker.grad(param) ## NEED TO FIGURE OUT DOUBLE GRAD IN FLUX
            push!(flat_grads, reshape(gg, length(gg)))
        else
            g = Tracker.grad(param)
            append!(flat_grads, reshape(g, length(g)))
        end
    end
    return flat_grads
end



"""
    flattenbatch(x::AbstractArray)
flatten a multi dimensional array to keep only the last dimension.
It returns a 2 dimensional array of size (flatten_dim, batch_size)
"""
function flattenbatch(x::AbstractArray)
    reshape(x, (:, size(x)[end]))
end

"""
    huber_loss(x, δ::Float64=1.0)
Compute the Huber Loss
"""
function huber_loss(x, δ::Float64=1.0)
    if abs(x) < δ
        return 0.5*x^2
    else
        return δ*(abs(x) - 0.5*δ)
    end
end

"""
    isrecurrent(m)
returns true if m contains a recurrent layer 
"""
function isrecurrent(m)
    for layer in m 
        if layer isa Flux.Recur 
            return true 
        end
    end
    return false
end

"""
    globalnorm(W)
returns the maximum absolute values in the gradients of W 
"""
function globalnorm(W)
    return maximum(maximum(abs.(w.grad)) for w in W)
end

"""
    batch_trajectories(s::AbstractArray, traj_length::Int64, batch_size::Int64)
converts multidimensional arrays into batches of trajectories to be process by a Flux recurrent model. 
It takes as input an array of dimension state_dim... x traj_length x batch_size
"""
function batch_trajectories(s::AbstractArray, traj_length::Int64, batch_size::Int64) 
    return Flux.batchseq([[s[axes(s)[1:end-2]..., j, i] for j=1:traj_length] for i=1:batch_size], 0)
end

""" 
    hiddenstates(m)
returns the hidden states of all the recurrent layers of a model 
""" 
function hiddenstates(m)
    return [l.state for l in m if l isa Flux.Recur]
end

"""
    sethiddenstates!(m, hs)
Given a list of hiddenstate, set the hidden state of each recurrent layer of the model m 
to what is in the list. 
The order of the list should match the order of the recurrent layers in the model.
"""
function sethiddenstates!(m, hs)
    i = 1
    for l in m
        if isa(l, Flux.Recur) 
            l.state = hs[i]
            i += 1
        end
    end
end