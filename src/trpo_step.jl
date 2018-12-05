    # def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    #     x = torch.zeros(b.size())
    #     r = b.clone()
    #     p = b.clone()
    #     rdotr = torch.dot(r, r)
    #     for i in range(nsteps):
    #         _Avp = Avp(p)
    #         alpha = rdotr / torch.dot(p, _Avp)
    #         x += alpha * p
    #         r -= alpha * _Avp
    #         new_rdotr = torch.dot(r, r)
    #         betta = new_rdotr / rdotr
    #         p = r + betta * p
    #         rdotr = new_rdotr
    #         if rdotr < residual_tol:
    #             break
    #     return x
    function conjugate_gradients(Avp, b, nsteps)
        x = zeros(size(b))
        r = deepcopy(b)
        p = deepcopy(b)
        rdotr = dot(vec(r),vec(r))
        resitual_tol = 1e-10

        for i = 1:nsteps
            _Avp = Avp(p)
            alpha = rdotr / dot(vec(p), vec(_Avp))
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = dot(vec(r), vec(r))
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < resitual_tol:
                break
        end

        return x
    end

    # def linesearch(model,
    #            f,
    #            x,
    #            fullstep,
    #            expected_improve_rate,
    #            max_backtracks=10,
    #            accept_ratio=.1):
    #     fval = f(True).data
    #     print("fval before", fval.item())
    #     for (_n_backtracks, stepfrac) in m:
    #         xnew = x + stepfrac * fullstep
    #         set_flat_params_to(model, xnew)
    #         newfval = f(True).data
    #         actual_improve = fval - newfval
    #         expected_improve = expected_improve_rate * stepfrac
    #         ratio = actual_improve / expected_improve
    #         print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

    #         if ratio.item() > accept_ratio and actual_improve.item() > 0:
    #             print("fval after", newfval.item())
    #             return True, xnew
    #     return False, x

    function linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks::Int64=10, accept_ratio::Float64=0.1)
        fval = f()
        backtrack_list = [(x, 0.5^x) for x in 1:max_backtracks]

        for (_n_backtracks, stepfrac) in backtrack_list
            xnew = x + stepfrac * fullstep
            set_flat_params_to(model, xnew)
            newfval = f()
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve

            if ratio > accept_ratio and actual_improve > 0:
                return true, xnew
        end
        return false, x
    end