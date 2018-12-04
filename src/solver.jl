@with_kw mutable struct TRPOSolver
    policy_network::Any = nothing # intended to be a flux model
    value_network::Any = nothing # intended to be a flux model 
    learning_rate::Float64 = 1e-4
    max_steps::Int64 = 1000
    batch_size::Int64 = 32
    train_freq::Int64 = 4
    eval_freq::Int64 = 500
    target_update_freq::Int64 = 500
    num_ep_eval::Int64 = 100
    recurrence::Bool = false
    eps_fraction::Float64 = 0.5
    eps_end::Float64 = 0.01
    evaluation_policy::Any = basic_evaluation
    exploration_policy::Any = linear_epsilon_greedy(max_steps, eps_fraction, eps_end)
    trace_length::Int64 = 40
    prioritized_replay::Bool = true
    prioritized_replay_alpha::Float64 = 0.6
    prioritized_replay_epsilon::Float64 = 1e-6
    prioritized_replay_beta::Float64 = 0.4
    buffer_size::Int64 = 1000
    max_episode_length::Int64 = 100
    train_start::Int64 = 200
    rng::AbstractRNG = MersenneTwister(0)
    logdir::String = ""
    save_freq::Int64 = 3000
    log_freq::Int64 = 100
    verbose::Bool = true
    l2_reg::Float64 = 1e-3
end

function POMDPs.solve(solver::TRPOSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.rng)
    return solve(solver, env)
end

function POMDPs.solve(solver::TRPOSolver, problem::POMDP)
    env = POMDPEnvironment(problem, rng=solver.rng)
    return solve(solver, env)
end

function POMDPs.solve(solver::TRPOSolver, env::AbstractEnvironment)
    # check reccurence 
    if isrecurrent(solver.policy_network) && !solver.recurrence
        throw("TRPOError: you passed in a recurrent model but recurrence is set to false")
    end

    # This is mainly just for ease of testing... will remove in final version
    if solver.policy_network == nothing
        solver.policy_network = PolicyNN(length(obs_dimensions(env)), 64, 64, n_actions(env))
    end

    if solver.value_network == nothing
        solver.value_network = ValueNN(length(obs_dimensions(env)), 64, 64, 1)
    end

    replay = initialize_replay_buffer(solver, env)
    value_network = solver.value_network
    policy = NNPolicy(env.problem, solver.policy_network, ordered_actions(env.problem), length(obs_dimensions(env)))
    optimizer = ADAM(Flux.params(value_network), solver.learning_rate)
    # start training
    reset!(policy)
    obs = reset(env)
    done = false
    episode_rewards = Float64[0.0]
    episode_steps = Float64[]
    saved_mean_reward = -Inf
    scores_eval = -Inf
    model_saved = false
    global_step = 0

    for k=1:solver.max_steps

    	# perform rollout
        num_samples = 0
    	while num_samples <= solver.batch_size
    		# reset Environment
    		obs = reset(env)
    		reset!(policy)

    		for t=1:10000
    			# select action
    			act, eps = exploration(solver.exploration_policy, policy, env, obs, global_step, solver.rng)
                ai = actionindex(env.problem, act)
                op, rew, done, info = step!(env, act)
                exp = TRPOExperience(obs, ai, rew, op, done)
                add_exp!(replay, exp)
                obs =  op # next_state = state
                episode_rewards[end] += rew
                if done
                    break
            num_samples += (t - 1)
            global_step += num_samples
    		end

    	end

        # train on experience from rollout
        hs = hiddenstates(policy_network) # Only important for recurrent networks
        loss_val, td_errors, grad_val = batch_train!(solver, env, optimizer, policy_network, value_network, replay)
        sethiddenstates!(value_network, hs) # Only important for recurrent networks

        if k%solver.eval_freq == 0
            scores_eval = evaluation(solver.evaluation_policy, 
                                 policy, env,                                  
                                 solver.num_ep_eval,
                                 solver.max_episode_length,
                                 solver.verbose)
        end

        if k%solver.log_freq == 0
            #TODO log the training perf somewhere (?dataframes/csv?)
            if  solver.verbose
                @printf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3e | Grad %2.3e \n",
                        k, solver.max_steps, eps, avg100_reward, loss_val, grad_val)
            end             
        end
        if k > solver.train_start && k%solver.save_freq == 0
            model_saved, saved_mean_reward = save_model(solver, policy_network, value_network, scores_eval, saved_mean_reward, model_saved)
        end

    end

    if model_saved
        if solver.verbose
            @printf("Restore model with eval reward %1.3f \n", saved_mean_reward)
            saved_model = BSON.load(solver.logdir*"policy_network.bson")[:policy_network]
            Flux.loadparams!(policy.network, saved_model)
        end
    end
    return policy
end

function initialize_replay_buffer(solver::TRPOSolver, env::AbstractEnvironment)
    # init and populate replay buffer
    if solver.recurrence
        replay = EpisodeReplayBuffer(env, solver.buffer_size, solver.batch_size, solver.trace_length)
    elseif solver.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.buffer_size, solver.batch_size)
    else
        replay = ReplayBuffer(env, solver.buffer_size, solver.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.train_start)
    return replay #XXX type unstable
end

# need to redefine loss
function loss(td)
    l = mean(huber_loss.(td))
    return l
end

function batch_train!(solver::TRPOSolver,
                      env::AbstractEnvironment,
                      policy_network,
                      value_network, 
                      s_batch, a_batch, r_batch, sp_batch, done_batch)

    ## define value loss function
    function get_value_loss(flat_params)
        set_flat_params_to(value_network, flat_params)

        _values = value_network(states)

        value_loss = mean((_values - targets).^2)

        # weight decay
        for param in params(value_network)
            value_loss += sum(param.^2)*solver.l2_reg
        back!(value_loss)
        return (Tracker.data(value_loss), Tracker.data(get_flat_grad_from(value_network))
    end

    # use L-BFGS optimizer (same as in the original paper implementation)
    res = optimize(get_value_loss, get_flat_params_from(value_network), LBFGS(), Optim.options(iterations=25)) # options used in corresponding pytorch implementation
    flat_params = minimizer(res) # gets the params that minimize the loss
    set_flat_params_to(value_network, flat_params)

    ## define policy loss function

    function get_policy_loss(flat_params)


    


        return loss_val, td_vals, grad_norm
    end

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
end



function batch_train!(solver::TRPOSolver,
                      env::AbstractEnvironment,
                      optimizer, 
                      value_network, 
                      target_q,
                      replay::ReplayBuffer)
    s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
    return batch_train!(solver, env, optimizer, value_network, target_q, s_batch, a_batch, r_batch, sp_batch, done_batch))
end

### TODO: Implement PrioritizedReplayBuffer for TRPO
### TODO: Implement EpisodeReplayBuffer for TRPO and Recurrent batch_train!


function save_model(solver::TRPOSolver, policy_network, value_network, scores_eval::Float64, saved_mean_reward::Float64, model_saved::Bool)
    if scores_eval >= saved_mean_reward
        policy_weights = Tracker.data.(params(policy_network))
        value_weights = Tracker.data.(params(value_network))
        bson(solver.logdir*"policy_network.bson", policy_network=policy_weights)
        bson(solver.logdir*"value_network.bson", value_network=value_weights)
        if solver.verbose
            @printf("Saving new model with eval reward %1.3f \n", scores_eval)
        end
        model_saved = true
        saved_mean_reward = scores_eval
    end
    return model_saved, saved_mean_reward
end

@POMDP_require solve(solver::TRPOSolver, mdp::Union{MDP, POMDP}) begin 
    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @req n_actions(::P)
    @subreq ordered_actions(mdp)
    if isa(mdp, POMDP)
        O = obstype(mdp)
        @req convert_o(::Type{AbstractArray}, ::O, ::P)
    else
        @req convert_s(::Type{AbstractArray}, ::S, ::P)
    end
    @req reward(::P,::S,::A,::S)
end