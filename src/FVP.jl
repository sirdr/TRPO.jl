using Flux
using Flux.Tracker

function fisher_vector_product(v)

    # if get_fim == nothing
    #     # Use direct method (Currently not supported in Julia)
    #     kl = get_kl(model)
    #     kl = mean(kl)

    #     # maybe make this a function if possible
    #     flat_grads = []
    #     for param in params(model)
    #         grads = Tracker.gradient(() -> kl, Params(param))
    #         g = grads[param]
    #         append!(flat_grads, reshape(g, length(g)))
    #     end

    #     kl_v = sum(flat_grads.*v)

    #     flat_grads_grads_kl = Float64[]
    #     for param in params(model)
    #         grads = Tracker.gradient(() -> kl_v, Params(param))
    #         g = Tracker.data(grads[param])
    #         append!(flat_grads_grads_kl, reshape(g, length(g)))
    #     end

    #     return flat_grads_grads_kl .+ v .* damping
    # else
    fim, action_probs = get_fim(model)
    t = param(ones(size(action_probs)))
    at = sum(action_probs .* t)

    Jt = []
    for param in params(model)
        grads = Tracker.gradient(() -> at, Params(param))
        g = grads[param]
        append!(Jt, reshape(g, length(g)))
    end

    Jtv = sum(Jt .* v)
    Jv = Tracker.gradient(() -> Jtv, Params(t))[t]

    MJv = fim .* Tracker.data(Jv)
    a_MJv = sum(action_probs .* MJv)
    #print("fim: $fim")
    JTMJv = []
    for param in params(model)
        grads = Tracker.gradient(() -> a_MJv, Params(param))
        g = Tracker.data(grads[param])
        append!(JTMJv, reshape(g, length(g)))
    end
    return JTMJv .+ v .* damping
end