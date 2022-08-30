#pragma diag_suppress 68, 1388, 1390, 1394, 20013, 20015 // disable nvcc warnings
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#define DIV_EPSILON 1e-5f

template <typename scalar_t>
__global__ void WeightedAverageForward(
    const int width,           //
    const int height,          //
    const int kernelWidth,     //
    const int halfKernelWidth, //
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Zero initialization
    output[0][y][x] = 0;
    output[1][y][x] = 0;
    output[2][y][x] = 0;

    // Iterate the kernel (v: y-axis, u: x-axis)
    scalar_t sumW = 0;
    for (int v = -halfKernelWidth; v <= halfKernelWidth; ++v)
    {
        int vy = v + y;
        if (vy < 0 || vy >= height)
            continue;
        for (int u = -halfKernelWidth; u <= halfKernelWidth; ++u)
        {
            int ux = u + x;
            if (ux < 0 || ux >= width)
                continue;

            // weight index
            int ind = (v + halfKernelWidth) * kernelWidth + (u + halfKernelWidth);

            output[0][y][x] += input[0][vy][ux] * weights[ind][y][x];
            output[1][y][x] += input[1][vy][ux] * weights[ind][y][x];
            output[2][y][x] += input[2][vy][ux] * weights[ind][y][x];
            sumW += weights[ind][y][x];
        }
    }
    scalar_t invSumW = 1 / fmaxf(sumW, DIV_EPSILON);
    output[0][y][x] *= invSumW;
    output[1][y][x] *= invSumW;
    output[2][y][x] *= invSumW;
}

// A wrapper function that launches the kernel.
torch::Tensor launchWeightedAverageForward(
    torch::Tensor input,  //
    torch::Tensor weights //
)
{
    const int width = input.size(2);
    const int height = input.size(1);
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    torch::Tensor output = torch::zeros_like(input);
    const uint32_t kernelWidth = (uint32_t)sqrt((float)weights.size(0));
    const uint32_t halfKernelWidth = (uint32_t)(kernelWidth / 2);

    // print("(launchWeightedAverage) input.scalar_type: %s\n", torch::toString(input.scalar_type()));
    // print("(launchWeightedAverage) kernelWidth: %u (half: %u)\n", kernelWidth, halfKernelWidth);
    // print("(launchWeightedAverage) weights.sizes: (%d, %d, %d)\n", weights.size(0), weights.size(1), weights.size(2));
    // print("(launchWeightedAverage) output.sizes: (%d, %d, %d)\n", output.size(0), output.size(1), output.size(2));

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "weighted_average_forward",
        (
            [&]
            {
                WeightedAverageForward<scalar_t><<<dimGrid, dimBlock>>>(
                    width, height,   // image size
                    kernelWidth,     // kernel width
                    halfKernelWidth, // half kernel width
                    input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
            }));
    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void WeightedAverageBackward(
    const int width,           //
    const int height,          //
    const int kernelWidth,     //
    const int halfKernelWidth, //
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> gradPrev,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> gradWeights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Iterate the kernel (v: y-axis, u: x-axis)
    for (int v = -halfKernelWidth; v <= halfKernelWidth; ++v)
    {
        int vy = v + y;
        if (vy < 0 || vy >= height)
            continue;
        for (int u = -halfKernelWidth; u <= halfKernelWidth; ++u)
        {
            int ux = u + x;
            if (ux < 0 || ux >= width)
                continue;

            // weight index
            int ind = (v + halfKernelWidth) * kernelWidth + (u + halfKernelWidth);

            gradWeights[ind][y][x] = input[0][vy][ux] * gradPrev[0][y][x]   //
                                     + input[1][vy][ux] * gradPrev[1][y][x] //
                                     + input[2][vy][ux] * gradPrev[2][y][x];
        }
    }
}

// A wrapper function that launches the kernel.
std::vector<torch::Tensor> launchWeightedAverageBackward(
    torch::Tensor input,   // [3, H, W]
    torch::Tensor weights, // [KxK, H, W]
    torch::Tensor gradPrev // [3, H, W]
)
{
    const int width = input.size(2);
    const int height = input.size(1);
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // [KxK, H, W]
    torch::Tensor gradWeights = torch::zeros(
        {weights.size(0) /*kernel*/, height, width}, // Shape
        torch::TensorOptions().device(torch::kCUDA)  // Device
    );
    const uint32_t kernelWidth = (uint32_t)sqrt((float)weights.size(0));
    const uint32_t halfKernelWidth = (uint32_t)(kernelWidth / 2);

    // print("(launchWeightedAverageBackward) input.scalar_type: %s\n", torch::toString(input.scalar_type()));
    // print("(launchWeightedAverageBackward) kernelWidth: %u (half: %u)\n", kernelWidth, halfKernelWidth);
    // print("(launchWeightedAverageBackward) gradWeights.sizes: (%d, %d, %d)\n", gradWeights.size(0), gradWeights.size(1), gradWeights.size(2));
    // print("(launchWeightedAverageBackward) gradPrev.sizes: (%d, %d, %d)\n", gradPrev.size(0), gradPrev.size(1), gradPrev.size(2));

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "weighted_average_backward",
        (
            [&]
            {
                WeightedAverageBackward<scalar_t><<<dimGrid, dimBlock>>>(
                    width, height,   // image size
                    kernelWidth,     // kernel width
                    halfKernelWidth, // half kernel width
                    input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradPrev.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    gradWeights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
            }));
    cudaDeviceSynchronize();

    // Return empty grad for input
    return {torch::Tensor(), gradWeights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &launchWeightedAverageForward, "Weighted average forward (CUDA)");
    m.def("backward", &launchWeightedAverageBackward, "Weighted average backward (CUDA)");
}