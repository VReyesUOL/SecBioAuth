use tfhe::core_crypto::gpu::{cuda_lwe_multi_bit_unsigned_scalar_comparison, CudaStreams};
use tfhe::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use tfhe::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use tfhe::core_crypto::gpu::lwe_multi_bit_bootstrap_key::CudaLweMultiBitBootstrapKey;
use tfhe::integer::gpu::ComparisonType;
use tfhe::shortint::MultiBitPBSParameters;

pub fn comparison(
    input: &CudaLweCiphertextList<u64>,
    scalar: u64,
    bsk: &CudaLweMultiBitBootstrapKey,
    ksk: &CudaLweKeyswitchKey<u64>,
    op: ComparisonType,
    params: MultiBitPBSParameters, stream: &CudaStreams) -> CudaLweCiphertextList<u64>
{
    cuda_lwe_multi_bit_unsigned_scalar_comparison(
        &input,
        &bsk,
        scalar,
        &ksk,
        op,
        params,
        stream,
    )
}