using POMDPModels
using POMDPSimulators
using Flux
using Random
using RLInterface
using Test

include("../src/TRPO.jl")
# using TRPO

function evaluate(mdp, solver, policy, rng, n_ep=100, max_steps=100)
    avg_r = 0.
    sim = RolloutSimulator(rng=rng, max_steps=max_steps)
    for i=1:n_ep
        solver.reset!(policy)
        avg_r += simulate(sim, mdp, policy)
    end
    return avg_r/=n_ep
end


# @testset "BabyPOMDP" begin 
# 	babyPOMDP = BabyPOMDP()

# 	function PolicyNN(a1_in, a1_out, a2_in, a2_out)
# 	    return Chain(x->TRPO.flattenbatch(x), Dense(a1_in, a1_out, tanh), Dense(a2_in, a2_out, tanh))
# 	end

# 	function ValueNN(a1_in, a1_out, a2_in, a2_out)
# 	    return Chain(x->TRPO.flattenbatch(x), Dense(a1_in, a1_out, tanh), Dense(a2_in, a2_out, tanh))
# 	end

# 	policy_network = PolicyNN(1, 16, 16, n_actions(tigerPOMDP))
# 	value_network = ValueNN(1, 16, 16, 1)
# 	rng = MersenneTwister(1)

# 	solver = TRPO.TRPOSolver(policy_network = policy_network, value_network=value_network, max_steps=10000, learning_rate=0.005, 
# 	                             eval_freq=2000,num_ep_eval=100,
# 	                             log_freq = 500, prioritized_replay=false,
# 	                             rng=rng)
	# using QMDP
	# solver = QMDPSolver()
	# babyPolicy = solve(solver, babyPOMDP)

# 	belief_updater = updater(babyPolicy) 

# 	history = simulate(HistoryRecorder(max_steps=10), babyPOMDP, babyPolicy, belief_updater)

# 	for (s, b, a, o) in eachstep(history, "sbao")
# 	    println("State was $s,")
# 	    println("belief was $b,")
# 	    println("action $a was taken,")
# 	    println("and observation $o was received.\n")
# 	end
# 	println("*******Discounted reward for baby was $(discounted_reward(history)).")
# end


@testset "TigerPOMDP" begin 
	rng = MersenneTwister(1)
    pomdp = TigerPOMDP(0.01, -1.0, 0.1, 0.8, 0.95);
    input_dims = reduce(*, size(convert_o(Vector{Float64}, first(observations(pomdp)), pomdp)))

	policy_network = PolicyNN(input_dims, 4, 4, n_actions(tigerPOMDP))
	value_network = ValueNN(input_dims, 4, 4, 1)

	solver = TRPO.TRPOSolver(policy_network = policy_network, value_network=value_network, max_steps=10000, learning_rate=0.005, 
	                             eval_freq=2000,num_ep_eval=100,
	                             log_freq = 500, prioritized_replay=false,
	                             rng=rng)

	tigerPolicy = solve(solver, tigerPOMDP)

	belief_updater = updater(tigerPolicy)

	history = simulate(HistoryRecorder(max_steps=10), tigerPOMDP, tigerPolicy, belief_updater)

	for (s, b, a, o) in eachstep(history, "sbao")
	    println("State was $s,")
	    println("belief was $b,")
	    println("action $a was taken,")
	    println("and observation $o was received.\n")
	end
	println("*******Discounted reward for Tiger was $(discounted_reward(history)).")
end


