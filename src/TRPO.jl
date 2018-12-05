module TRPO

using Random
using StatsBase
using Printf
using Parameters
using Flux
using BSON
using POMDPs
using POMDPModelTools
using POMDPPolicies
using RLInterface
using Optim
using LinearAlgebra

export TRPOSolver,
       AbstractNNPolicy,
       NNPolicy,
       TRPOExperience,
    
       # helpers
       flattenbatch,
       huber_loss,
       isrecurrent,
       batch_trajectories

include("helpers.jl")
include("policy.jl")
include("exploration_policy.jl")
include("evaluation_policy.jl")
include("experience_replay.jl")
include("prioritized_experience_replay.jl")
include("episode_replay.jl")
include("solver.jl")
include("running_state.jl")
include("models.jl")

end # module DeepQLearning