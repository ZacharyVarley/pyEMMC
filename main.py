"""

This file implements the Monte Carlo simulation of electron scattering from:

Joy, David C. Monte Carlo modeling for electron microscopy and microanalysis.
Vol. 9. Oxford University Press, 1995.

The code is based on the OpenCL implementation from the following file:

https://github.com/EMsoft-org/EMsoftOO/blob/develop/opencl/EMMC.cl

Unlike EMMC.cl, this PyOpenCL implementation uses atomics to accumulate
the histogram directly on the GPU.

"""

import numpy as np
import pyopencl as cl
import pyopencl.array
from typing import Dict, Any
import os
import time

# Constants
PI = 3.14159265359
OPENCL_KERNEL = """
#define PI 3.14159265359f

typedef struct {
    float x;
    float y;
} LambertStruct;

LambertStruct rosca_lambert(float4 pt) {
    float factor = sqrt(fmax(2.0f * (1.0f - fabs(pt.z)), 0.0f));
    float big = fabs(pt.y) <= fabs(pt.x) ? pt.x : pt.y;
    float sml = fabs(pt.y) <= fabs(pt.x) ? pt.y : pt.x;
    float sign = big < 0 ? -1.0f : 1.0f;
    float simpler_term = sign * factor * (2.0f / sqrt(8.0f));
    float arctan_term = sign * factor * atan2(sml * sign, fabs(big)) * (2.0f * sqrt(2.0f) / PI);
    return (LambertStruct){
        fabs(pt.y) <= fabs(pt.x) ? simpler_term : arctan_term,
        fabs(pt.y) <= fabs(pt.x) ? arctan_term : simpler_term
    };
}

uint4 lfsr113_Bits(uint4 z) {
    uint b;
    b = ((z.x << 6) ^ z.x) >> 13; z.x = ((z.x & 4294967294U) << 18) ^ b;
    b = ((z.y << 2) ^ z.y) >> 27; z.y = ((z.y & 4294967288U) << 2) ^ b;
    b = ((z.z << 13) ^ z.z) >> 21; z.z = ((z.z & 4294967280U) << 7) ^ b;
    b = ((z.w << 3) ^ z.w) >> 12; z.w = ((z.w & 4294967168U) << 13) ^ b;
    return z;
}

__kernel void bse_sim_mc(
    __global uint* accumulator,
    __global uint4* seeds,
    const float n_trials_per_electron,
    const float starting_E_keV,
    const int n_exit_energy_bins,
    const int n_exit_direction_bins,
    const int n_exit_depth_bins,
    const float binsize_exit_energy,
    const float binsize_exit_depth,
    const float atom_num,
    const float unit_cell_density_rho,
    const float atomic_weight_A,
    const int n_max_steps,
    const float sigma,
    const float omega,
    const int depth_mode
) {
    int gid = get_global_id(0);
    float sigma_rad = sigma * PI / 180.0f;
    float omega_rad = omega * PI / 180.0f;
    float mean_ionization_pot_J = ((9.76f * atom_num) + (58.5f * pow(atom_num, -0.19f))) * 1.0e-3f;

    // Precompute constants
    const float const_0_00785 = -0.00785f * (atom_num / atomic_weight_A);
    const float const_5_21 = 5.21f * 602.2f * pow(atom_num, 2.0f);
    const float const_3_4e_3 = 3.4e-3f * pow(atom_num, 0.66667f);
    const float const_1e7_over_rho = 1.0e7f * atomic_weight_A / unit_cell_density_rho;
    const float rand_factor = 2.32830643708079737543146996187e-10f;

    uint4 seed = seeds[gid];

    for (int i = 0; i < n_trials_per_electron; i++) {
        float4 current_direction = (float4)(
            sin(sigma_rad) * cos(omega_rad),
            sin(sigma_rad) * sin(omega_rad),
            cos(sigma_rad),
            0.0f
        );
        current_direction.xyz /= sqrt(dot(current_direction.xyz, current_direction.xyz));

        float energy = starting_E_keV;

        for (int step = 0; step < n_max_steps; step++) {
            float energy_inv = 1.0f / energy;
            float alpha = const_3_4e_3 * energy_inv;
            float sigma_E = const_5_21 * energy_inv * energy_inv *
                            (4.0f * PI / (alpha * (1.0f + alpha))) *
                            pow((511.0f + energy) / (1024.0f + energy), 2);
            float mean_free_path_nm = const_1e7_over_rho / sigma_E;

            seed = lfsr113_Bits(seed);
            float rand_step = (float)(seed.x ^ seed.y ^ seed.z ^ seed.w) * rand_factor;
            float step_nm = -mean_free_path_nm * log(rand_step);

            float de_ds = const_0_00785 * energy_inv * log((1.166f * energy / mean_ionization_pot_J) + 0.9911f);

            seed = lfsr113_Bits(seed);
            float rand_phi = (float)(seed.x ^ seed.y ^ seed.z ^ seed.w) * rand_factor;
            float phi = acos(1.0f - ((2.0f * alpha * rand_phi) / (1.0f + alpha - rand_phi)));

            seed = lfsr113_Bits(seed);
            float rand_psi = (float)(seed.x ^ seed.y ^ seed.z ^ seed.w) * rand_factor;
            float psi = 2.0f * PI * rand_psi;

            float4 c_old = current_direction;
            float4 c_new;

            float cos_phi = cos(phi);
            float sin_phi = sin(phi);
            if (fabs(c_old.z) > 0.99999f) {
                float cos_psi = cos(psi);
                float sin_psi = sin(psi);
                c_new = (float4)(
                    sin_phi * cos_psi,
                    sin_phi * sin_psi,
                    (c_old.z > 0 ? 1.0f : -1.0f) * cos_phi,
                    c_old.w
                );
            } else {
                float dsq = sqrt(1.0f - c_old.z * c_old.z);
                float dsqi = 1.0f / dsq;
                float cos_psi = cos(psi);
                float sin_psi = sin(psi);
                c_new = (float4)(
                    sin_phi * (c_old.x * c_old.z * cos_psi - c_old.y * sin_psi) * dsqi + c_old.x * cos_phi,
                    sin_phi * (c_old.y * c_old.z * cos_psi + c_old.x * sin_psi) * dsqi + c_old.y * cos_phi,
                    -sin_phi * cos_psi * dsq + c_old.z * cos_phi,
                    c_old.w
                );
            }

            c_new.xyz /= sqrt(dot(c_new.xyz, c_new.xyz));

            float escape_depth = fabs(c_new.w / c_new.z);
            c_new.w += step_nm * c_new.z;
            energy += step_nm * unit_cell_density_rho * de_ds;

            current_direction = c_new;

            if (energy <= 0) {
                energy = 0;
                break;
            } else if (current_direction.w < 0) {
                int exit_depth_index = depth_mode == 0 ?
                    (int)(escape_depth / binsize_exit_depth) :
                    (int)(log(escape_depth + 1) / binsize_exit_depth);
                int exit_energy_index = (int)((starting_E_keV - energy) * (1.0f / binsize_exit_energy));
                LambertStruct lambert = rosca_lambert(current_direction);
                int2 exit_direction_index = (int2)(
                    (int)((lambert.x * 0.499999f + 0.5f) * n_exit_direction_bins),
                    (int)((lambert.y * 0.499999f + 0.5f) * n_exit_direction_bins)
                );

                if (exit_energy_index >= 0 && exit_energy_index < n_exit_energy_bins &&
                    exit_depth_index >= 0 && exit_depth_index < n_exit_depth_bins &&
                    exit_direction_index.x >= 0 && exit_direction_index.x < n_exit_direction_bins &&
                    exit_direction_index.y >= 0 && exit_direction_index.y < n_exit_direction_bins) {
                    int index = exit_energy_index * (n_exit_depth_bins * n_exit_direction_bins * n_exit_direction_bins) +
                                exit_depth_index * (n_exit_direction_bins * n_exit_direction_bins) +
                                exit_direction_index.x * n_exit_direction_bins +
                                exit_direction_index.y;
                    atomic_inc(&accumulator[index]);
                }
                break;
            }
        }
    }

    seeds[gid] = seed;
}
"""

class OpenCLMonteCarlo:
    def __init__(self, params: Dict[str, Any]):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, OPENCL_KERNEL).build()
        self.n_electrons = params["n_electrons"]
        self._initialize_buffers(params)

    def _initialize_buffers(self, params: Dict[str, Any]):
        histogram_shape = (
            params["n_exit_energy_bins"],
            params["n_exit_depth_bins"],
            params["n_exit_direction_bins"],
            params["n_exit_direction_bins"],
        )
        self.accumulator = cl.array.zeros(self.queue, histogram_shape, dtype=np.uint32)
        self.seeds = cl.array.to_device(
            self.queue,
            np.random.randint(0, 2**32, size=(self.n_electrons, 4), dtype=np.uint32),
        )

    def run_simulation(self, params: Dict[str, Any]) -> np.ndarray:
        self.program.bse_sim_mc(
            self.queue,
            (self.n_electrons,),
            None,
            self.accumulator.data,
            self.seeds.data,
            np.float32(params["n_trials_per_electron"]),
            np.float32(params["starting_E_keV"]),
            np.int32(params["n_exit_energy_bins"]),
            np.int32(params["n_exit_direction_bins"]),
            np.int32(params["n_exit_depth_bins"]),
            np.float32(params["binsize_exit_energy"]),
            np.float32(params["binsize_exit_depth"]),
            np.float32(params["atom_num"]),
            np.float32(params["unit_cell_density_rho"]),
            np.float32(params["atomic_weight_A"]),
            np.int32(params["n_max_steps"]),
            np.float32(params["sigma"]),
            np.float32(params["omega"]),
            np.int32(0 if params["depth_mode"] == "linear" else 1),
        )

        return self.accumulator.get()

# Example usage
if __name__ == "__main__":
    material_params = {
        "atom_num": 13.0,  # Aluminum
        "unit_cell_density_rho": 2.7,  # Aluminum density
        "atomic_weight_A": 26.98,  # Aluminum atomic weight
    }

    n_extra = 2
    simulation_params = {
        "n_electrons": 1000000,
        "n_trials_per_electron": 10, # increase to do more sims
        "starting_E_keV": 20.0,
        "n_exit_energy_bins": 30 + n_extra, # it is recommended to have 2 faux bins
        "n_exit_depth_bins": 80,
        "n_exit_direction_bins": 501,
        "depth_mode": "log",  # or "linear"
        "n_max_steps": 1000,
        "sigma": 70.0,
        "omega": 0.0,
        "binsize_exit_energy": 1.0,
        "binsize_exit_depth": 0.1,
    }

    # Set the PYOPENCL_CTX environment variable
    os.environ["PYOPENCL_CTX"] = "0" # use the first detected OpenCL device

    params = {**material_params, **simulation_params}
    mc = OpenCLMonteCarlo(params)

    start = time.time()
    histogram = mc.run_simulation(params)

    print(f"Simulation time: {time.time() - start:.2f} seconds")
  
    # remove the placeholder energy bins (changes estimated yield)
    histogram = histogram[n_extra:]
  
    print("Simulation completed.")
    print(f"Total backscattered electrons: {np.sum(histogram)}")
