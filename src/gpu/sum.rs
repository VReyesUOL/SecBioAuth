use tfhe::core_crypto::gpu::{cuda_lwe_multi_bit_sum, CudaStreams};
use tfhe::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use tfhe::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use tfhe::core_crypto::gpu::lwe_multi_bit_bootstrap_key::CudaLweMultiBitBootstrapKey;
use tfhe::core_crypto::prelude::LweCiphertextCount;
use tfhe::shortint::MultiBitPBSParameters;

pub fn sum(
    input: &mut CudaLweCiphertextList<u64>, blocks: usize, num_cts: usize,
    bsk: &CudaLweMultiBitBootstrapKey,
    ksk: &CudaLweKeyswitchKey<u64>,
    params: MultiBitPBSParameters, stream: &CudaStreams) -> CudaLweCiphertextList<u64> {
    let mut result = CudaLweCiphertextList::new(
        params.glwe_dimension.to_equivalent_lwe_dimension(params.polynomial_size),
        LweCiphertextCount(blocks),
        params.ciphertext_modulus,
        &stream,
    );
    cuda_lwe_multi_bit_sum(
        input,
        &mut result,
        &bsk,
        &ksk,
        blocks,
        num_cts,
        params,
        stream,
    );
    result
}