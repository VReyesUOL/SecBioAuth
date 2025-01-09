use std::fs;
use itertools::{izip, Itertools};
use rand::Rng;
use tfhe::core_crypto::prelude::UnsignedInteger;
use crate::common::config::{Config, DATA_PATH, LOOKUP_TABLES_FOLDER, PATH_SEPARATOR, QBIN_SUFFIX, TABLE_PREFIX};
use crate::common::util::{decompose_to_base, flip_vectors, length_in_base};


pub fn load_and_offset_helr_tables(config: &Config) -> (i64, Vec<Vec<Vec<u64>>>) {
    let helr_filename_basepath = [DATA_PATH, LOOKUP_TABLES_FOLDER, config.data_set_name, TABLE_PREFIX].join(PATH_SEPARATOR);
    let helr_tables = read_helr_tables(helr_filename_basepath.as_str(), config.num_tables);
    offset_helr_tables(&helr_tables)
}

pub fn get_probe_and_template(config: &Config, index: usize) -> (Vec<u64>, Vec<u64>) {
    let qbin_filename = format!("{}{}.csv", config.data_set_name, QBIN_SUFFIX);
    let qbins_path = [DATA_PATH, LOOKUP_TABLES_FOLDER, config.data_set_name, qbin_filename.as_str()].join(PATH_SEPARATOR);
    let qbins = read_qbins(qbins_path.as_str());

    let feature_vector_filename = format!("{}.csv", config.data_set_name);
    let feature_vector_path = [DATA_PATH, feature_vector_filename.as_str()].join(PATH_SEPARATOR);
    let template = read_feature_vector(feature_vector_path.as_str(), index);
    let probe = distort_feature_vector(&template);

    let quantized_template = quantize_vector(&template, &qbins);
    let quantized_probe = quantize_vector(&probe, &qbins);

    (quantized_probe, quantized_template)
}

pub fn get_min_decomp_base(tables: &[Vec<Vec<u64>>]) -> u64 {
    let dimension = tables.first().expect("No tables").len();
    let bits = dimension.ceil_ilog2();
    1u64 << (bits / 2)
}

pub fn get_lut_output_indices(tables: &[Vec<Vec<u64>>], decomp_base: u64) -> (Vec<u64>, Vec<usize>, usize) {
    let max_values = tables.iter().map(|table|
        table.first().and_then(|row| row.first()).expect("Table is empty").clone()
    ).collect_vec();
    let max_total_sum: u64 = max_values.iter().sum();

    let decomp_lengths: Vec<usize> = max_values.iter()
        .map(|max_value| length_in_base(*max_value, decomp_base))
        .collect_vec();
    let sum_decomp_length = length_in_base(max_total_sum, decomp_base);

    let total_entries: usize = decomp_lengths.iter().sum();

    let mut lut_output_indices = Vec::with_capacity(total_entries);
    decomp_lengths.iter().enumerate().for_each(|(idx, len)| {
        (0..*len).for_each(|v| lut_output_indices.push((idx * sum_decomp_length + v) as u64))
    });

    (lut_output_indices, decomp_lengths, sum_decomp_length)
}

fn decompose_template(tables: &[Vec<Vec<u64>>], template: &[u64], base: u64) -> Vec<Vec<Vec<u64>>> {
    izip!(tables, template).map(|(table, template_idx)| {
        let max_value = table.first().and_then(|row| row.first()).expect("Could not get max value");
        let length = length_in_base(*max_value, base);

        let row = table.get(*template_idx as usize).expect("Indexing error");
        row.iter().map(|v| decompose_to_base(*v, base, length)).collect_vec()
    }).collect_vec()
}

pub(crate) fn make_row_based_luts(tables: &[Vec<Vec<u64>>], template: &[u64], base: u64) -> Vec<Vec<Vec<u64>>> {
    let decomposed = decompose_template(tables, template, base);
    decomposed.into_iter().map(|inner| flip_vectors(inner)).collect_vec()
}

fn read_qbins(path: &str) -> Vec<f64> {
    let qbins_csv = fs::read_to_string(path).expect("Unable to read file");
    qbins_csv.split(",").map(|v| v.parse::<f64>().expect("Could not parse entry")).collect()
}

fn read_helr_tables(path: &str, n_tables: usize) -> Vec<Vec<Vec<i64>>>{
    (0..n_tables).map(|idx| {
        let table_path = format!("{path}{idx}.csv");
        let file_contents = fs::read_to_string(table_path).expect("Unable to read file");
        file_contents.lines().map(|line|
            line.split(",").map(|v| v.parse::<i64>().expect("could not parse entry")).collect()
        ).collect()
    }).collect()
}

fn offset_helr_tables(helr_tables: &[Vec<Vec<i64>>]) -> (i64, Vec<Vec<Vec<u64>>>) {
    let mut offset: i64 = 0;
    let offset_helr_tables = helr_tables.iter().map(|table| {
        let local_offset = table.first().and_then(|v| v.last()).expect("Unable to index helr table");
        offset += local_offset;
        table.into_iter().map(|row| {
            row.into_iter().map(|v| (v - local_offset) as u64).collect()
        }).collect()
    }).collect();
    (offset, offset_helr_tables)
}

fn read_feature_vector(path: &str, id: usize) -> Vec<f64> {
    let dataset_csv = fs::read_to_string(path).expect("Unable to read file");
    let mut dataset_lines = dataset_csv.lines();
    let id_line = dataset_lines.nth(id).expect("Error indexing feature vector entries");
    let values = id_line.split(",").skip(1).map(|v| v.parse::<f64>().expect("could not parse entry")).collect();
    values
}

fn distort_feature_vector(feat_vec: &[f64]) -> Vec<f64>{
    let mut rng = rand::thread_rng();
    feat_vec.iter().cloned().map(|v| v + rng.gen_range(-0.01..=0.01)).collect()
}

fn quantize_vector(feat_vec: &[f64], qbins: &[f64]) -> Vec<u64> {
    let l = qbins.len();
    feat_vec.iter().map(|f| {
        qbins.iter().position(|v| *f <= *v).unwrap_or(l) as u64
    }).collect()
}