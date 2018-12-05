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
    gamma::Float64 = 0.995
    tau::Float64 = 0.97
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

    values = value_network(s_batch)

    returns = zeros(1, solver.batch_size)
    deltas = zeros(1, solver.batch_size)
    advantages = zeros(1, solver.batch_size)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0

    for i in size(r_batch)[end]:-1:1
        returns[:,i] = r_batch[:,i] + gamma * prev_return * done_batch[:,i]
        deltas[:,i] = r_batch[:,i] + gamma * prev_value * done_batch[:,i] - Tracker.data(values)[:,i]
        advantages[:,i] = deltas[:,i] + gamma * tau * prev_advantage * done_batch[:,i]

        prev_return = returns[1, i]
        prev_value = Tracker.data(values)[1, i]
        prev_advantage = advantages[1, i]
    end

    targets = param(returns)

    ## define value loss function
    function get_value_loss(flat_params)
        set_flat_params_to!(value_network, flat_params)
        _values = value_network(s_batch)
        value_loss = mean((_values - targets).^2)
        # weight decay
        for param in params(value_network)
            value_loss += sum(param.^2)*l2_reg
        end
        #Flux.back!(value_loss)
        for param in params(value_network)
            grads = Tracker.gradient(() -> value_loss, Params(param))
        end
        #Tracker.data(get_flat_grad_from(value_network))
        return Tracker.data(value_loss)
    end

    function g!(storage, x)
        _values = value_network(s_batch)
        value_loss = mean((_values - targets).^2)
        # weight decay
        for param in params(value_network)
            value_loss += sum(param.^2)*l2_reg
        end
        #Flux.back!(value_loss)
        flat_grads = Float64[]
        for param in params(value_network)
            grads = Tracker.gradient(() -> value_loss, Params(param))
            g = Tracker.data(grads[param])
            append!(flat_grads, reshape(g, length(g)))
        end
        for i in 1:length(flat_grads)
            storage[i] = flat_grads[i]
        end
    end

    res = optimize(get_value_loss, g!, get_flat_params_from(value_network), LBFGS(), Optim.Options(iterations=25)) # options used in corresponding pytorch implementation
    flat_params = Optim.minimizer(res) # gets the params that minimize the loss
    set_flat_params_to!(value_network, flat_params)

    # Begin: TODO - figure out return of policy network
    actions = policy_network(s_batch) 
    # End: TODO

    # update advantage
    advantages = (advantages .- mean(advantages))./std(advantages)

    actions
    fixed_softmax_loss = logsoftmax!(a_batch)
    fixed_softmax_loss = fixed_softmax_loss[]

    ## define policy loss function
    function get_policy_loss()

        actions = policy_net(s_batch)
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))

    end


    ## define KL loss for discrete distributions
    function get_kl()
        actions = policy_net(s_batch)

    end
    
    


        return loss_val, td_vals, grad_norm
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