use itertools::Itertools;
use tfhe::core_crypto::entities::LweCiphertextListOwned;
use tfhe::core_crypto::gpu::glwe_ciphertext_list::CudaGlweCiphertextList;
use tfhe::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use tfhe::core_crypto::gpu::CudaStreams;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64, PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M64};
use tfhe::shortint::{CarryModulus, MessageModulus, MultiBitPBSParameters, ShortintParameterSet};

pub fn get_params_multi_bit_gpu(total_bits: u64) -> MultiBitPBSParameters {
    match total_bits {
        4 => PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64,
        8 => PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M64,
        _ => panic!("Not implemented for {} bits", total_bits)
    }
}

pub fn encode_encrypt_list(values: &[u64], delta: u64, sk: LweSecretKeyView<u64>, noise_distribution: DynamicDistribution<u64>, mut encryption_generator: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>, params: &ShortintParameterSet) -> LweCiphertextListOwned<u64> {
    let msg_count = values.len();
    let pt_count = PlaintextCount(msg_count);
    let ct_count = LweCiphertextCount(msg_count);
    let mod_sup = params.message_modulus().0 * params.carry_modulus().0;

    let mut pt_list = PlaintextList::new(0, pt_count);
    pt_list.iter_mut().zip(values)
        .for_each(|(dst, value)| *dst.0 = (*value) % mod_sup as u64 * delta);

    let mut output = LweCiphertextList::new(
        0,
        sk.lwe_dimension().to_lwe_size(),
        ct_count,
        params.ciphertext_modulus(),
    );

    par_encrypt_lwe_ciphertext_list(
        &sk,
        &mut output,
        &pt_list,
        noise_distribution,
        &mut encryption_generator,
    );

    output
}

pub fn encode_encrypt_list_cuda(values: &[u64], delta: u64, sk: LweSecretKeyView<u64>, noise_distribution: DynamicDistribution<u64>, encryption_generator: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>, params: &ShortintParameterSet, streams: &CudaStreams) -> CudaLweCiphertextList<u64> {
    let output = encode_encrypt_list(values, delta, sk, noise_distribution, encryption_generator, params);
    CudaLweCiphertextList::from_lwe_ciphertext_list(&output, streams)
}

pub fn decrypt_decode_list(input_cts: LweCiphertextListView<u64>, delta: u64, sk: LweSecretKeyView<u64>) -> Vec<u64> {
    let pt_count = PlaintextCount(input_cts.lwe_ciphertext_count().0);
    let mut output_pt_list = PlaintextList::new(0, pt_count);
    decrypt_lwe_ciphertext_list(&sk, &input_cts, &mut output_pt_list);

    //The bit before the message
    let rounding_bit = delta >> 1;

    output_pt_list.iter().map(|pt| {
        let decrypted_u64 = *pt.0;
        //compute the rounding bit
        let rounding = (decrypted_u64 & rounding_bit) << 1;

        (decrypted_u64.wrapping_add(rounding)) / delta
    }).collect_vec()
}

pub fn decrypt_decode_list_cuda(input_cts: &CudaLweCiphertextList<u64>, delta: u64, sk: LweSecretKeyView<u64>, streams: &CudaStreams) -> Vec<u64> {
    let lwe_cts = input_cts.to_lwe_ciphertext_list(&streams);
    decrypt_decode_list(lwe_cts.as_view(), delta, sk)
}

pub fn new_ct_list(ct_count: usize, lwe_size: LweSize, params: &ShortintParameterSet) -> LweCiphertextListOwned<u64> {
    LweCiphertextList::new(
        0,
        lwe_size,
        LweCiphertextCount(ct_count),
        params.ciphertext_modulus(),
    )
}

pub fn new_ct_list_cuda(ct_count: usize, lwe_size: LweSize, params: &ShortintParameterSet, streams: &CudaStreams) -> CudaLweCiphertextList<u64> {
    let output = new_ct_list(ct_count, lwe_size, params);
    CudaLweCiphertextList::from_lwe_ciphertext_list(&output, streams)
}

pub fn fill_accumulator<F, C>(
    accumulator: &mut GlweCiphertext<C>,
    polynomial_size: PolynomialSize,
    glwe_size: GlweSize,
    message_modulus: MessageModulus,
    carry_modulus: CarryModulus,
    f: F,
) -> u64
where
    C: ContainerMut<Element=u64>,
    F: Fn(u64) -> u64, //, <C as crate::core_crypto::commons::traits::container::Container>::Element: UnsignedInteger
{
    assert_eq!(accumulator.polynomial_size(), polynomial_size);
    assert_eq!(accumulator.glwe_size(), glwe_size);

    let mut accumulator_view = accumulator.as_mut_view();

    accumulator_view.get_mut_mask().as_mut().fill(0);

    // Modulus of the msg contained in the msg bits and operations buffer
    let modulus_sup = message_modulus.0 * carry_modulus.0;

    // N/(p/2) = size of each block
    let box_size = polynomial_size.0 / modulus_sup;

    // Value of the shift we multiply our messages by
    let delta = (1_u64 << 63) / (message_modulus.0 * carry_modulus.0) as u64;

    let mut body = accumulator_view.get_mut_body();
    let accumulator_u64 = body.as_mut();

    // Tracking the max value of the function to define the degree later
    let mut max_value = 0;

    for i in 0..modulus_sup {
        let index = i * box_size;
        let f_eval = f(i as u64);
        max_value = max_value.max(f_eval);
        accumulator_u64[index..index + box_size].fill(f_eval * delta);
    }

    let half_box_size = box_size / 2;

    // Negate the first half_box_size coefficients
    for a_i in accumulator_u64[0..half_box_size].iter_mut() {
        *a_i = (*a_i).wrapping_neg();
    }

    // Rotate the accumulator
    accumulator_u64.rotate_left(half_box_size);

    max_value
}

pub fn make_accumulator_list<F>(fs: &[F], params: &ShortintParameterSet) -> GlweCiphertextListOwned<u64>
where
    F: Fn(u64) -> u64,
{
    let count = fs.len();
    let mut glwe_list = GlweCiphertextList::new(0, params.glwe_dimension().to_glwe_size(), params.polynomial_size(), GlweCiphertextCount(count), params.ciphertext_modulus());
    glwe_list.iter_mut().zip(fs).for_each(|(mut ct, f)| {
        fill_accumulator(&mut ct, params.polynomial_size(), params.glwe_dimension().to_glwe_size(), params.message_modulus(), params.carry_modulus(), f);
    });
    glwe_list
}

pub fn make_encrypted_accumulator_list<F>(
    fs: &[F],
    params: &ShortintParameterSet,
    glwe_secret_key: GlweSecretKeyView<u64>,
    noise: DynamicDistribution<u64>,
    generator: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>,
) -> GlweCiphertextListOwned<u64>
where
    F: Fn(u64) -> u64,
{
    let count = fs.len();
    let mut glwe_list = GlweCiphertextList::new(0, params.glwe_dimension().to_glwe_size(), params.polynomial_size(), GlweCiphertextCount(count), params.ciphertext_modulus());
    glwe_list.iter_mut().zip(fs).for_each(|(mut ct, f)| {
        fill_accumulator(&mut ct, params.polynomial_size(), params.glwe_dimension().to_glwe_size(), params.message_modulus(), params.carry_modulus(), f);
        encrypt_glwe(&glwe_secret_key, &mut ct, noise, generator);
    });
    glwe_list
}

pub fn make_accumulator_list_cuda<F>(fs: &[F], params: &ShortintParameterSet, streams: &CudaStreams) -> CudaGlweCiphertextList<u64>
where
    F: Fn(u64) -> u64,
{
    let glwe_list = make_accumulator_list(fs, params);
    CudaGlweCiphertextList::from_glwe_ciphertext_list(&glwe_list, streams)
}

pub fn make_encrypted_accumulator_list_cuda<F>(
    fs: &[F],
    params: &ShortintParameterSet,
    glwe_secret_key: GlweSecretKeyView<u64>,
    noise: DynamicDistribution<u64>,
    generator: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>,
    streams: &CudaStreams
) -> CudaGlweCiphertextList<u64>
where
    F: Fn(u64) -> u64,
{
    let glwe_list = make_encrypted_accumulator_list(fs, params, glwe_secret_key, noise, generator);
    CudaGlweCiphertextList::from_glwe_ciphertext_list(&glwe_list, streams)
}

pub fn encrypt_glwe(
    glwe_secret_key: &GlweSecretKeyView<u64>,
    mut glwe_ct: &mut GlweCiphertextMutView<u64>,
    noise: DynamicDistribution<u64>,
    generator: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>,
) {
    encrypt_glwe_ciphertext_assign(
        &glwe_secret_key,
        &mut glwe_ct,
        noise,
        generator,
    )
}