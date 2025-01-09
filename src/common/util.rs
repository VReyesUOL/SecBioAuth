use itertools::Itertools;

pub fn decompose_to_base(mut value: u64, base: u64, blocks: usize) -> Vec<u64> {
    (0..blocks).map(|_| {
        let res = value % base;
        value = value / base;
        res
    }).collect()
}

pub fn length_in_base(value: u64, base: u64) -> usize {
    ((value + 1) as f64).log(base as f64).ceil() as usize
}


pub fn flip_vectors<T>(vecs: Vec<Vec<T>>) -> Vec<Vec<T>> {
    let max_l = vecs.iter().map(|v| v.len()).max().expect("vectors are empty");
    let mut new_vecs: Vec<Vec<T>> = Vec::with_capacity(max_l);
    for _ in 0..max_l {
        new_vecs.push(Vec::new());
    }
    vecs.into_iter().for_each(|vec| {
        vec.into_iter().enumerate().for_each(|(idx, v)| {
            new_vecs[idx].push(v);
        })
    });
    new_vecs
}

//TESTING
pub fn luts_to_closures(luts: Vec<Vec<u64>>) -> Vec<Box<dyn Fn(u64) -> u64>>
{
    luts.into_iter().map(|lut| {
        Box::new(move |x: u64| lut[x as usize]) as Box<dyn Fn(u64) -> u64>
    }).collect_vec()
}
