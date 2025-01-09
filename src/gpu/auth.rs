use crate::common::config::Config;
use crate::common::data;
use crate::common::tfhe_utils::{decrypt_decode_list_cuda, encode_encrypt_list_cuda, get_params_multi_bit_gpu, make_encrypted_accumulator_list_cuda};
use crate::common::util::{decompose_to_base, luts_to_closures};
use crate::gpu;
use itertools::{izip, repeat_n, Itertools};
use std::time::Instant;
use tfhe::integer::gpu::ComparisonType;
use tfhe::shortint::{MultiBitPBSParameters, PBSParameters, ShortintParameterSet};

fn get_data(config: &Config, idx: usize) -> (u64, Vec<u64>, Vec<Vec<Vec<u64>>>, Vec<u64>, usize, MultiBitPBSParameters) {
    let (offset, helr_tables) = data::load_and_offset_helr_tables(&config);
    let (probe, template) = data::get_probe_and_template(&config, idx);
    let decomp_base = data::get_min_decomp_base(&helr_tables);
    let luts = data::make_row_based_luts(&helr_tables, &template, decomp_base);
    let (lut_output_indices, decomp_lengths, sum_block_len) = data::get_lut_output_indices(&helr_tables, decomp_base);
    let repeated_probes = izip!(probe.iter(), decomp_lengths).map(|(p, len)| repeat_n(*p, len)).flatten().collect_vec();
    let params = get_params_multi_bit_gpu(decomp_base);

    let expected: u64 = izip!(helr_tables.iter(), probe.iter(), template.iter()).map(|(table, idx_t, idx_p)| table[*idx_t as usize][*idx_p as usize]).sum();
    let expected_decomp = decompose_to_base(expected, decomp_base, sum_block_len);

    println!("Running {}:", config.data_set_name);
    println!("Decomposition base: {}", decomp_base);
    println!("Luts: {}", lut_output_indices.len());
    println!("Sum size: {} x {} = {} blocks", sum_block_len, probe.len(), sum_block_len * probe.len());
    println!("Expected: {} [{:?}]", expected, expected_decomp);

    (
        (config.threshold - offset) as u64,
        repeated_probes,
        luts,
        lut_output_indices,
        sum_block_len,
        params
    )
}

pub fn auth(config: Config) {
    let (threshold, repeated_probes, luts, output_indices, sum_block_len, params) = get_data(&config, 0);
    let short_params = ShortintParameterSet::new_pbs_param_set(PBSParameters::MultiBitPBS(params));

    let num_cts = luts.len();
    let pbs_out_blocks = num_cts * sum_block_len;
    let num_total_blocks = output_indices.len();
    let flat_luts = luts.into_iter().flatten().collect_vec();

    println!("Key gen...");
    let (
        stream,
        mut encryption_generator,
        lwe_secret_key,
        glwe_secret_key,
        multi_bit_bsk_gpu,
        d_key_switching_key,
        delta
    ) = gpu::keygen::genkeys_multibit_cuda(params);

    println!("Encrypt...");
    let lwe_ciphertext_in_gpu = encode_encrypt_list_cuda(
        &repeated_probes,
        delta,
        lwe_secret_key.as_view(),
        params.lwe_noise_distribution,
        &mut encryption_generator,
        &short_params,
        &stream,
    );

    let functions = luts_to_closures(flat_luts);
    let glwe_luts_in_gpu = make_encrypted_accumulator_list_cuda(
        &functions,
        &short_params,
        glwe_secret_key.as_view(),
        params.glwe_noise_distribution,
        &mut encryption_generator,
        &stream,
    );

    let start = Instant::now();
    //println!("PBS...");
    let mut pbs_res = gpu::encrypted_pbs(
        lwe_ciphertext_in_gpu,
        glwe_luts_in_gpu,
        output_indices,
        num_total_blocks,
        pbs_out_blocks,
        glwe_secret_key.as_lwe_secret_key().lwe_dimension().to_lwe_size(),
        &short_params,
        &multi_bit_bsk_gpu,
        &stream,
    );

    //println!("Sum...");
    let sum_res = gpu::sum(
        &mut pbs_res,
        sum_block_len,
        num_cts,
        &multi_bit_bsk_gpu,
        &d_key_switching_key,
        params,
        &stream,
    );

    //println!("Comparison...");
    let comp_res = gpu::comparison(
        &sum_res,
        threshold,
        &multi_bit_bsk_gpu,
        &d_key_switching_key,
        ComparisonType::GE,
        params,
        &stream,
    );
    let elapsed = start.elapsed();

    println!("Decrypting...");
    let result = decrypt_decode_list_cuda(
        &comp_res,
        delta,
        glwe_secret_key.as_lwe_secret_key(),
        &stream,
    );
    let clear_sum = decrypt_decode_list_cuda(&sum_res, delta, glwe_secret_key.as_lwe_secret_key(), &stream);
    println!("Comparison Result:");
    println!("Got: {:?} [{:?} >= {}]", result, clear_sum, threshold);
    println!("Total time needed: {}s", elapsed.as_secs_f64());
}
