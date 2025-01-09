use itertools::Itertools;
use tfhe::core_crypto::gpu::{cuda_multi_bit_programmable_bootstrap_lwe_ciphertext, CudaStreams};
use tfhe::core_crypto::gpu::glwe_ciphertext_list::CudaGlweCiphertextList;
use tfhe::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use tfhe::core_crypto::gpu::lwe_multi_bit_bootstrap_key::CudaLweMultiBitBootstrapKey;
use tfhe::core_crypto::gpu::vec::CudaVec;
use tfhe::core_crypto::prelude::LweSize;
use tfhe::shortint::ShortintParameterSet;
use crate::common::tfhe_utils::{make_accumulator_list_cuda, new_ct_list_cuda};
use crate::common::util::luts_to_closures;

pub fn pbs(
    cts_in: CudaLweCiphertextList<u64>,
    luts: Vec<Vec<u64>>,
    output_indices: Vec<u64>,
    num_total_blocks: usize,
    num_output_blocks: usize,
    lwe_size: LweSize,
    short_params: &ShortintParameterSet,
    bsk: &CudaLweMultiBitBootstrapKey,
    stream: &CudaStreams,
) -> CudaLweCiphertextList<u64> {
    let functions = luts_to_closures(luts);

    let h_indexes = (0..num_total_blocks as u64).collect_vec();
    let mut d_output_indexes = unsafe { CudaVec::<u64>::new_async(num_total_blocks, &stream, 0) };
    let mut d_input_indexes = unsafe { CudaVec::<u64>::new_async(num_total_blocks, &stream, 0) };
    let mut d_lut_indexes = unsafe { CudaVec::<u64>::new_async(num_total_blocks, &stream, 0) };
    unsafe {
        d_output_indexes.copy_from_cpu_async(output_indices.as_ref(), &stream, 0);
        d_input_indexes.copy_from_cpu_async(h_indexes.as_ref(), &stream, 0);
        d_lut_indexes.copy_from_cpu_async(h_indexes.as_ref(), &stream, 0);
    }
    stream.synchronize();

    let accumulator_gpu = make_accumulator_list_cuda(&functions, &short_params, &stream);

    let mut out_pbs_ct_gpu = new_ct_list_cuda(
        num_output_blocks,
        lwe_size,
        &short_params,
        &stream,
    );

    cuda_multi_bit_programmable_bootstrap_lwe_ciphertext(
        &cts_in,
        &mut out_pbs_ct_gpu,
        &accumulator_gpu,
        &d_lut_indexes,
        &d_output_indexes,
        &d_input_indexes,
        &bsk,
        &stream,
    );

    out_pbs_ct_gpu
}


pub fn encrypted_pbs(
    cts_in: CudaLweCiphertextList<u64>,
    accumulator_gpu: CudaGlweCiphertextList<u64>,
    output_indices: Vec<u64>,
    num_total_blocks: usize,
    num_output_blocks: usize,
    lwe_size: LweSize,
    short_params: &ShortintParameterSet,
    bsk: &CudaLweMultiBitBootstrapKey,
    stream: &CudaStreams,
) -> CudaLweCiphertextList<u64> {
    //let functions = luts_to_closures(luts);

    let h_indexes = (0..num_total_blocks as u64).collect_vec();
    let mut d_output_indexes = unsafe { CudaVec::<u64>::new_async(num_total_blocks, &stream, 0) };
    let mut d_input_indexes = unsafe { CudaVec::<u64>::new_async(num_total_blocks, &stream, 0) };
    let mut d_lut_indexes = unsafe { CudaVec::<u64>::new_async(num_total_blocks, &stream, 0) };
    unsafe {
        d_output_indexes.copy_from_cpu_async(output_indices.as_ref(), &stream, 0);
        d_input_indexes.copy_from_cpu_async(h_indexes.as_ref(), &stream, 0);
        d_lut_indexes.copy_from_cpu_async(h_indexes.as_ref(), &stream, 0);
    }
    stream.synchronize();

    let mut out_pbs_ct_gpu = new_ct_list_cuda(
        num_output_blocks,
        lwe_size,
        &short_params,
        &stream,
    );

    cuda_multi_bit_programmable_bootstrap_lwe_ciphertext(
        &cts_in,
        &mut out_pbs_ct_gpu,
        &accumulator_gpu,
        &d_lut_indexes,
        &d_output_indexes,
        &d_input_indexes,
        &bsk,
        &stream,
    );

    out_pbs_ct_gpu
}