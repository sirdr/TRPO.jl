
function conjugate_gradients(Avp, b, nsteps)
    x = zeros(size(b))
    r = deepcopy(b)
    p = deepcopy(b)
    rdotr = dot(vec(r),vec(r))
    resitual_tol = 1e-10

    for i = 1:nsteps
        if count(n -> abs(n) < 1e-20, p) == length(p)
            return x
        end
        _Avp = Avp(p)
        alpha = rdotr / dot(vec(p), vec(_Avp))
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = dot(vec(r), vec(r))
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < resitual_tol
            break
        end
    end

    return x
end

function linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks::Int64=10, accept_ratio::Float64=0.1)
    fval = f(model)
    backtrack_list = [(x, 0.5^x) for x in 1:max_backtracks]

    for (_n_backtracks, stepfrac) in backtrack_list
        xnew = x + stepfrac * fullstep
        set_flat_params_to!(model, xnew)
        newfval = f(model)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio && actual_improve > 0
            return true, xnew
        end
    end
    return false, x
end

function trpo_step(model, get_loss, get_kl, max_kl, damping, get_fim)

    loss = get_loss(model)

    flat_grads_loss = Float64[]
    for param in params(model)
        grads = Tracker.gradient(() -> loss, Params(param))
        g = Tracker.data(grads[param])
        append!(flat_grads_loss, reshape(g, length(g)))
    end

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

        JTMJv = []
        for param in params(model)
            grads = Tracker.gradient(() -> a_MJv, Params(param))
            g = Tracker.data(grads[param])
            append!(JTMJv, reshape(g, length(g)))
        end
        return JTMJv .+ v .* damping
    end

    step_direction = conjugate_gradients(fisher_vector_product, -1 .*flat_grads_loss, 10)

    fvp = fisher_vector_product(step_direction)

    shs = 0.5 .*sum(step_direction .* fvp)

    lagrange_multiplier = broadcast(sqrt, shs/max_kl)
    fullstep = step_direction./lagrange_multiplier[1]

    negdot_stepdir = -1 .*sum(step_direction.*flat_grads_loss)

    first_lm = lagrange_multiplier[1]
    grad_norm = norm(flat_grads_loss)

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     negdot_stepdir ./ first_lm)
    set_flat_params_to!(model, new_params)

    return loss, new_params, grad_norm
end