// Known-good probe kernel used by PodHealthTracker (spec §6.8, §10.3).
// Self-contained vec_add program: the kernel launches on a tiny fixed
// input, synchronizes, and asserts the output element-wise. Exit 0 on
// success, 1 on any CUDA error or value mismatch. The probe MUST be
// deterministic and free of I/O so `PodHealthTracker.run_probe_if_needed`
// can invoke it as an opaque subprocess.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" __global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static int fail(const char* msg, cudaError_t err) {
    std::fprintf(stderr, "probe_error: %s (%s)\n", msg, cudaGetErrorString(err));
    return 1;
}

int main(int /*argc*/, char** /*argv*/) {
    // Tiny, fixed input — small N keeps the probe under 1 ms on any GPU.
    constexpr int N = 1024;
    float h_a[N];
    float h_b[N];
    float h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 0.25f;
        h_c[i] = 0.0f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaError_t err = cudaMalloc(&d_a, N * sizeof(float));
    if (err != cudaSuccess) return fail("cudaMalloc A", err);
    err = cudaMalloc(&d_b, N * sizeof(float));
    if (err != cudaSuccess) return fail("cudaMalloc B", err);
    err = cudaMalloc(&d_c, N * sizeof(float));
    if (err != cudaSuccess) return fail("cudaMalloc C", err);

    err = cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail("memcpy A H2D", err);
    err = cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail("memcpy B H2D", err);
    err = cudaMemset(d_c, 0, N * sizeof(float));
    if (err != cudaSuccess) return fail("memset C", err);

    const int block = 128;
    const int grid = (N + block - 1) / block;
    vec_add<<<grid, block>>>(d_a, d_b, d_c, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) return fail("launch", err);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return fail("synchronize", err);

    err = cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return fail("memcpy C D2H", err);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Verify: h_c[i] == h_a[i] + h_b[i]. The operands are exact in fp32,
    // so equality is strict — any mismatch means the pod cannot run even
    // a trivial kernel and must be quarantined.
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            std::fprintf(
                stderr,
                "probe_mismatch: i=%d got=%.9g expected=%.9g\n",
                i,
                static_cast<double>(h_c[i]),
                static_cast<double>(expected));
            return 1;
        }
    }
    return 0;
}
