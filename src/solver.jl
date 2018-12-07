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
    max_kl::Float64 = 0.01
    damping::Float64 = 0.1
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

    replay = initialize_replay_buffer(solver, env)
    policy = NNPolicy(env.problem, solver.policy_network, ordered_actions(env.problem), length(obs_dimensions(env)))
    # start training
    reset!(policy)
    obs = reset(env)
    done = false
    all_rewards = Float64[]
    episode_rewards = Float64[0.0]
    episode_steps = Float64[]
    saved_mean_reward = -Inf
    scores_eval = -Inf
    model_saved = false
    global_step = 0
    optimizer = ADAM(Flux.params(solver.value_network), solver.learning_rate)

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
                end
                num_samples += 1
            end
            global_step += num_samples
            append!(all_rewards, episode_rewards / num_samples)
    	end

        # train on experience from rollout
        hs = hiddenstates(solver.policy_network) # Only important for recurrent networks
        loss_val, grad_val = batch_train!(solver, env, solver.policy_network, solver.value_network, replay, optimizer)
        sethiddenstates!(solver.value_network, hs) # Only important for recurrent networks

        if k%solver.eval_freq == 0
            scores_eval = evaluation(solver.evaluation_policy, 
                                 policy, env,                                  
                                 solver.num_ep_eval,
                                 solver.max_episode_length,
                                 solver.verbose)
        end

        if k%solver.log_freq == 0
            #TODO log the training perf somewhere (?dataframes/csv?)
            # println(all_rewards)
            avg100_reward = sum(all_rewards) / length(all_rewards)
            # println(avg100_reward)
            #println("eps: $eps")
            if  solver.verbose
                @printf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3e | Grad %2.3e \n",
                        k, solver.max_steps, global_step, avg100_reward, Tracker.data(loss_val), Tracker.data(grad_val))
            end             
        end
        if k > solver.train_start && k%solver.save_freq == 0
            model_saved, saved_mean_reward = save_model(solver, solver.policy_network, solver.value_network, scores_eval, saved_mean_reward, model_saved)
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
                      old_policy_network,
                      value_network, 
                      s_batch, a_batch, r_batch, sp_batch, done_batch, optimizer)

    new_policy_network = deepcopy(old_policy_network)

    values = value_network(s_batch)

    mask = broadcast(x -> abs(Int(x) - 1), done_batch)

    returns = zeros(solver.batch_size)
    deltas = zeros(solver.batch_size)
    advantages = zeros(solver.batch_size)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0

    for i in size(r_batch)[end]:-1:1
        returns[i] = r_batch[i] + solver.gamma * prev_return * mask[i]
        deltas[i] = r_batch[i] + solver.gamma * prev_value * mask[i] - Tracker.data(values)[1,i]
        advantages[i] = deltas[i] + solver.gamma * solver.tau * prev_advantage * mask[i]

        prev_return = returns[i]
        prev_value = Tracker.data(values)[1,i]
        prev_advantage = advantages[i]
    end

    targets = param(returns)

    ## define value loss function
    function get_value_loss(flat_params)
        set_flat_params_to!(value_network, flat_params)
        _values = value_network(s_batch)
        value_loss = mean((_values[1,:] - targets).^2)
        # weight decay
        for param in params(value_network)
            value_loss += sum(param.^2)*solver.l2_reg
        end
        #Flux.back!(value_loss)
        #Tracker.data(get_flat_grad_from(value_network))
        return Tracker.data(value_loss)
    end

    function g!(storage, x)
        _values = value_network(s_batch)
        value_loss = mean((_values[1,:] - targets).^2)
        # weight decay
        for param in params(value_network)
            value_loss += sum(param.^2)*solver.l2_reg
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

    # res = optimize(get_value_loss, g!, get_flat_params_from(value_network), LBFGS(), Optim.Options(iterations=25)) # options used in corresponding pytorch implementation
    # flat_params = Optim.minimizer(res) # gets the params that minimize the loss
    # set_flat_params_to!(value_network, flat_params)

    ## Uncomment below for vanilla backprop
    _values = value_network(s_batch)
    value_loss = mean((_values[1,:] - targets).^2)
    # weight decay
    for param in params(value_network)
        value_loss += sum(param.^2)*solver.l2_reg
    end
    Flux.back!(value_loss)
    optimizer()

    actions = old_policy_network(s_batch) 

    # update advantage
    advantages = (advantages .- mean(advantages))./std(advantages)

    n_actions = size(actions)[1]
    action_mask = [Int(a_batch[div(i-1, n_actions)+1] == (i-1)%n_actions+1) for (i, a) in enumerate(actions)]

    old_log_softmax = NNlib.logsoftmax(actions)

    old_log_prob = sum(action_mask .*old_log_softmax, dims=1)

    ## define policy loss function
    function get_policy_loss(net)
        #new_actions = policy_network(s_batch)
        new_actions = net(s_batch)
        new_log_softmax = NNlib.logsoftmax(new_actions)
        new_log_prob = sum(action_mask.*new_log_softmax, dims=1)
        policy_loss = -1 .* advantages .* broadcast(exp, (new_log_prob - old_log_prob))
        return mean(policy_loss)
    end


    ## define KL loss for discrete distributions
    function get_kl(net)
        #new_actions = policy_network(s_batch)
        new_actions = net(s_batch)
        new_log_softmax = NNlib.logsoftmax(new_actions)
        kl = broadcast(exp, new_log_softmax).*(old_log_softmax .- new_log_softmax)
        kl = sum(kl, dims=1)
        return kl
    end

    # Calculate Fisher Information Matrix (FIM) of each input
    # Note: the FIM is diagonal for each input with elements of 1/prob(action)
    # which allows us to store it as a vector for each input in the batch
    # with dimension equal to the action space
    function get_fim(net)
        new_actions = net(s_batch)
        #new_softmax = NNlib.softmax(new_actions)
        #new_softmax = broadcast(exp, new_log_softmax)
        new_exp = broadcast(exp, new_actions)
        new_sum = sum(new_exp, dims=1)
        new_softmax = new_exp ./ new_sum
        fim = broadcast(inv, new_softmax)
        return fim, new_softmax
    end

    loss_val, new_params_flat, grad_val = trpo_step(new_policy_network, get_policy_loss, get_kl, solver.max_kl, solver.damping, get_fim)
    set_flat_params_to!(old_policy_network, new_params_flat)
    return loss_val, grad_val
end



function batch_train!(solver::TRPOSolver,
                      env::AbstractEnvironment, 
                      old_policy_network, 
                      value_network,
                      replay::ReplayBuffer, optimizer)
    s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
    return batch_train!(solver, env, old_policy_network, value_network, s_batch, a_batch, r_batch, sp_batch, done_batch, optimizer)
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