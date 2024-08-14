use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

// # Brief
// * Root Mean Square normalization, calculate y = x * w / sqrt(sum(x^2) / len(x) + epsilon)
// # Arg
// * y: 2D output tensor (in-place)
// * x: 2D input tensor
// * w: 1D weight tensor
// * epsilon: small value to avoid division by zero
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let w_dim = w.shape().len();
    let x_dim = x.shape().len();
    assert!(w_dim == 1);
    assert!(x_dim == 2); // TODO: FIX ME
    assert_eq!(x.shape()[x_dim - 1], w.shape()[0]);
    let _y = unsafe { y.data_mut() };
    for i in 0..x.shape()[0] {
        let mut ele_sqrt = 0.0;
        let len = w.shape()[0];
        for j in 0..x.shape()[x.shape().len() - 1] {
            ele_sqrt += x.data()[i * len + j] * x.data()[i * len + j];
        }
        ele_sqrt = (ele_sqrt / len as f32).sqrt();
        let lower = ele_sqrt + epsilon;
        println!("ele_sqrt: {}, lower: {}", ele_sqrt, lower);
        for j in 0..x.shape()[x.shape().len() - 1] {
            _y[i * len + j] = x.data()[i * len + j] * w.data()[j] / lower;
        }
    }
}

// # Brief
// * Sigmoid Linear Unit (SiLU) activation function, calculate y = y * x / (1 + exp(-x))
// # Arg
// * y: output tensor (in-place)
// * x: input tensor
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    for i in 0..len {
        let x = _x[i];
        _y[i] *= x / (1. + (-x).exp());
    }
}

// # Brief
// * Matrix multiplication with transposed B, calculate C = beta * C + alpha * A @ B^T
// # Arg
// * c: output tensor (in-place)
// * beta: scaling factor for c
// * a: input tensor
// * b: input weight tensor
// * alpha: scaling factor for a @ b^T
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    assert!(c.shape().len() == 2); // TODO: only support two-dimensional matrix
    assert!(a.shape().len() == 2);
    assert!(b.shape().len() == 2);
    let y_num = c.shape()[1];
    let x_num = c.shape()[0];
    let k_num = a.shape()[1];
    let _c = unsafe { c.data_mut() };
    for x in 0..x_num {
        let row = &a.data()[x * k_num..(x + 1) * k_num];
        for y in 0..y_num {
            let col = &b.data()[y * k_num..(y + 1) * k_num];
            let sum = row.iter().zip(col.iter()).map(|(a, b)| a * b).sum::<f32>();
            _c[x * y_num + y] = beta * _c[x * y_num + y] + alpha * sum;
        }
    }
}

// # Brief
// * Element-wise addition of two matrices, calculate A = A + B
// Arg
// * A: output tensor (in-place)
// * B: input tensor
pub fn mat_add(A: &mut Tensor<f32>, B: &Tensor<f32>) {
    assert_eq!(A.shape(), B.shape());
    assert!(A.shape().len() <= 2, "only support 1D or 2D matrix");
    let shape = A.shape().clone();
    let shape_dim = A.shape().len();
    let a_data = unsafe { A.data_mut() };
    if shape_dim == 1 {
        for i in 0..shape[0] {
            a_data[i] += B.data()[i];
        }
    }
    if shape_dim == 2 {
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                a_data[i * shape[1] + j] += B.data()[i * shape[1] + j];
            }
        }
    }
}

// # Brief
// * alloc a new tensor, concat two tensors along a given axis
// All data will be deep copied
// Arg
// * A: output tensor (in-place)
// * B: input tensor
pub fn alloc_concat(A: &Tensor<f32>, B: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    assert!(A.shape().len() == B.shape().len());
    // check whther concat is avaliable in current axis
    for i in 0..A.shape().len() {
        if i != axis {
            assert!(A.shape()[i] == B.shape()[i]);
        };
    }
    let mut new_shape = A.shape().clone();
    new_shape[axis] += B.shape()[axis];
    // suppose A.shape() == [2, 3, 4], B.shape() == [2, 1, 4]
    // memory view will be:
    // [A[0,:,:],B[0,:,:]]
    // [A[1,:,:],B[1,:,:]]
    // So we need to split A and B into 2 chunks and concat them
    let A_chunk_size: usize = (&A.shape()[axis..]).iter().product();
    let B_chunk_size: usize = (&B.shape()[axis..]).iter().product();
    println!(
        "A_chunk_size: {}, B_chunk_size: {}",
        A_chunk_size, B_chunk_size
    );
    let A_data_chunk = A.data().chunks(A_chunk_size);
    let B_data_chunk = B.data().chunks(B_chunk_size);
    assert!(A_data_chunk.len() == B_data_chunk.len());
    let new_data = A_data_chunk
        .zip(B_data_chunk)
        .flat_map(|(chunk1, chunk2)| {
            let mut result = Vec::new();
            result.extend_from_slice(chunk1);
            result.extend_from_slice(chunk2);
            result
        })
        .collect();
    Tensor::new(new_data, &new_shape)
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_alloc_concat() {
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let mut c = alloc_concat(&a, &b, 1);
    assert!(c.close_to(
        &Tensor::<f32>::new(
            vec![1., 2., 3., 1., 2., 3., 4., 5., 6., 4., 5., 6.],
            &vec![2, 6]
        ),
        1e-3
    ));
    let c = alloc_concat(&a, &b, 0);
    assert!(c.close_to(
        &Tensor::<f32>::new(
            vec![1., 2., 3., 4., 5., 6.,1.,2.,3., 4., 5., 6.],
            &vec![4, 3]
        ),
        1e-3
    ));
}
