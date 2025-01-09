use tfhe::core_crypto::commons::generators::DeterministicSeeder;
use tfhe::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use tfhe::core_crypto::gpu::lwe_multi_bit_bootstrap_key::CudaLweMultiBitBootstrapKey;
use tfhe::core_crypto::gpu::CudaStreams;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::MultiBitPBSParameters;

pub fn genkeys_multibit_cuda(parameters: MultiBitPBSParameters) -> (CudaStreams, EncryptionRandomGenerator<ActivatedRandomGenerator>, LweSecretKeyOwned<u64>, GlweSecretKeyOwned<u64>, CudaLweMultiBitBootstrapKey, CudaLweKeyswitchKey<u64>, u64) {
    //Init seeders and rnd generators
    let mut root_seeder = new_seeder();

    let mut seeder =
        DeterministicSeeder::<ActivatedRandomGenerator>::new(root_seeder.seed());

    let mut secret_generator: SecretRandomGenerator<ActivatedRandomGenerator> = SecretRandomGenerator::new(seeder.seed());
    let mut encryption_generator = EncryptionRandomGenerator::new(
        seeder.seed(),
        &mut seeder,
    );

    // Gen Client keys
    // generate the lwe secret key
    let lwe_secret_key = allocate_and_generate_new_binary_lwe_secret_key(
        parameters.lwe_dimension,
        &mut secret_generator,
    );

    // generate the rlwe secret key
    let glwe_secret_key = allocate_and_generate_new_binary_glwe_secret_key(
        parameters.glwe_dimension,
        parameters.polynomial_size,
        &mut secret_generator,
    );

    // Gen Server Keys
    //Generate CudaStream
    let gpu_idx = 0;
    let streams = CudaStreams::new_single_gpu(gpu_idx);

    // Generate a regular keyset and convert to the GPU
    let h_bootstrap_key: LweMultiBitBootstrapKeyOwned<u64> =
        par_allocate_and_generate_new_lwe_multi_bit_bootstrap_key(
            &lwe_secret_key.as_view(),
            &glwe_secret_key,
            parameters.pbs_base_log,
            parameters.pbs_level,
            parameters.grouping_factor,
            parameters.glwe_noise_distribution,
            parameters.ciphertext_modulus,
            &mut encryption_generator,
        );

    let d_bootstrapping_key = CudaLweMultiBitBootstrapKey::from_lwe_multi_bit_bootstrap_key(
        &h_bootstrap_key,
        &streams,
    );

    // Creation of the key switching key
    let h_key_switching_key = allocate_and_generate_new_lwe_keyswitch_key(
        &glwe_secret_key.as_lwe_secret_key(),
        &lwe_secret_key.as_view(),
        parameters.ks_base_log,
        parameters.ks_level,
        parameters.lwe_noise_distribution,
        parameters.ciphertext_modulus,
        &mut encryption_generator,
    );

    let d_key_switching_key =
        CudaLweKeyswitchKey::from_lwe_keyswitch_key(&h_key_switching_key, &streams);

    //The delta is the one defined by the parameters
    let delta = (1_u64 << 63)
        / (parameters.message_modulus.0 * parameters.carry_modulus.0)
        as u64;

    (
        streams,
        encryption_generator,
        lwe_secret_key,
        glwe_secret_key,
        d_bootstrapping_key,
        d_key_switching_key,
        delta
    )
}