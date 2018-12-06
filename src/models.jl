using Flux
using Flux.Tracker

function PolicyNN(a1_in, a1_out, a2_in, a2_out)
	return Chain(x->flattenbatch(x), Dense(a1_in, a1_out, tanh), Dense(a2_in, a2_out, tanh))
end

function ValueNN(a1_in, a1_out, a2_in, a2_out)
	return Chain(x->flattenbatch(x), Dense(a1_in, a1_out, tanh), Dense(a2_in, a2_out, tanh))
end
