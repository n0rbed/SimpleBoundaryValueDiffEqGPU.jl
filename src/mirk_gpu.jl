using CUDA, KernelAbstractions, BVProblemLibrary, SciMLBase
using SimpleBoundaryValueDiffEq: __extract_details, alg_stage, SimpleMIRK4, constructSimpleMIRK

@kernel function reskernel!(residual, y, mesh, len_mesh, stages_flat, n_stage, c, v, b, x, dt, size_u)
    i = @index(Global)
    if i <= (len_mesh - 1)
        column_stage = i % size_u == 0 ? size_u : i % size_u

        for r in 1:n_stage

            @inbounds x_temp = mesh[i] + c[r] * dt
            @inbounds y_temp = (1 - v[r]) * y[i] + v[r] * y[i + size_u]

            if r > 1
                for j = 1:(r-1)
                    y_temp += x[r,j] * stages_flat[(column_stage)+(j-1)*size_u]
                end
                y_temp *= dt
            end

            # TO DEAL WITH
            # prob.f(stages_flat[r], y_temp, prob.p, x_temp)
        end

        sum_bstages = 0
        for j = 1:n_stage
            sum_bstages += b[j] * stages_flat[(column_stage)+(j-1)*size_u]
        end

        residual[i] = y[i + size_u] - y[i] - dt * sum_bstages
    end
end


###### Example of how the input to the kernel looks like ######

alg = SimpleMIRK4()

prob = eval(Symbol("prob_bvp_linear_1"))
dt = 0.1

N = Int(cld(prob.tspan[2] - prob.tspan[1], dt))
mesh = collect(range(prob.tspan[1], prob.tspan[2], length = N + 1))
iip = SciMLBase.isinplace(prob)
pt = prob.problem_type
c, v, b, x = constructSimpleMIRK(alg)

# usual CPU data, copied from SimpleBoundaryValueDiffEq
stage = alg_stage(alg)
M, u0, guess = __extract_details(prob, N)
resid = [similar(u0) for _ in 1:(N + 1)]
y = [similar(u0) for _ in 1:(N + 1)]
discrete_stages = [similar(u0) for _ in 1:stage]

# GPU data
size_u = size(u0)[1]
gpu_resid = CuArray(zeros(Float32, size_u * N+1))
y_flat = CuArray(zeros(Float32, size_u * N+1))

n_stage = stage
gpu_stages = CuArray(zeros(Float32, n_stage*size_u))

gpu_c, gpu_v, gpu_b, gpu_x = (CuArray(Float32.(n)) for n in [c,v,b,x])
gpu_mesh = CuArray(mesh)
len_mesh = length(mesh)

k = reskernel!(get_backend(gpu_resid))
k(gpu_resid, y_flat, gpu_mesh, len_mesh, gpu_stages, n_stage,
    gpu_c, gpu_v, gpu_b, gpu_x, dt, size_u, ndrange=len_mesh-1)


