using POMDPModels
using POMDPSimulators
using Flux
using Random
using RLInterface
using Test
using TRPO

include("test_env.jl")

function evaluate(mdp, policy, rng, n_ep=100, max_steps=100)
    avg_r = 0.
    sim = RolloutSimulator(rng=rng, max_steps=max_steps)
    for i=1:n_ep
        TRPO.reset!(policy)
        avg_r += simulate(sim, mdp, policy)
    end
    return avg_r/=n_ep
end

@testset "vanilla TRPO" begin 
    rng = MersenneTwister(1)
    mdp = TestMDP((5,5), 4, 6)
    #model = ValueNN(100, 8, 8, n_actions(mdp))#Chain(x->flattenbatch(x), Dense(100, 8, tanh), Dense(8, n_actions(mdp)))

    # This is mainly just for ease of testing... will remove in final version
    function PolicyNN(a1_in, a1_out, a2_in, a2_out)
        return Chain(x->flattenbatch(x), Dense(a1_in, a1_out, tanh), Dense(a2_in, a2_out, tanh))
    end

    function ValueNN(a1_in, a1_out, a2_in, a2_out)
        return Chain(x->flattenbatch(x), Dense(a1_in, a1_out, tanh), Dense(a2_in, a2_out, tanh))
    end

    policy_network = PolicyNN(100, 64, 64, n_actions(mdp))
    value_network = ValueNN(100, 64, 64, 1)

    solver = TRPOSolver(policy_network = policy_network, value_network=value_network, max_steps=10000, learning_rate=0.005, 
                                 eval_freq=2000,num_ep_eval=100,
                                 log_freq = 500, prioritized_replay=false,
                                 rng=rng)

    policy = solve(solver, mdp)
    r_basic = evaluate(mdp, policy, rng)
    @test r_basic >= 1.5
end