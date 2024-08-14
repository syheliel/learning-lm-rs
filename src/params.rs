use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // safetensor.tensor(tensor_name);

        let get_tensor = |name: &str| {
            let tensorView = safetensor.tensor(name).unwrap();
            assert!(tensorView.dtype() == safetensors::Dtype::F32);
            let tensor_data = tensorView
                .data()
                .chunks_exact(core::mem::size_of::<f32>())
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            Tensor::new(tensor_data, &tensorView.shape().to_vec())
        };

        // for each layer, get the corresponding tensor and return a tensor list [Tensor_1, Tensor_2, ..., Tensor_n]
        // the tensor name pattern is like "model.layers.{}.input_layernorm.weight"
        macro_rules! get_tensor_vec {
            ($name_pattern:literal) => {
                (0..config.num_hidden_layers)
                    .map(|i| get_tensor(&format!($name_pattern, i)))
                    .collect()
            };
        }

        // [overall tensor list]
        // - `lm_head.weight`
        // - `model.norm.weight`
        // [layernorm]
        // - `model.layers.0.input_layernorm.weight`
        // - `model.layers.0.post_attention_layernorm.weight`
        // [mlp]
        // - `model.layers.0.mlp.down_proj.weight`
        // - `model.layers.0.mlp.gate_proj.weight`
        // - `model.layers.0.mlp.up_proj.weight`
        // [self_attn]
        // - `model.layers.0.self_attn.k_proj.weight`
        // - `model.layers.0.self_attn.o_proj.weight`
        // - `model.layers.0.self_attn.q_proj.weight`
        // - `model.layers.0.self_attn.v_proj.weight`
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            lm_head: get_tensor("lm_head.weight"),
            rms_out_w: get_tensor("model.norm.weight"),

            rms_att_w: get_tensor_vec!("model.layers.{}.input_layernorm.weight"),
            rms_ffn_w: get_tensor_vec!("model.layers.{}.post_attention_layernorm.weight"),

            w_up: get_tensor_vec!("model.layers.{}.mlp.up_proj.weight"),
            w_gate: get_tensor_vec!("model.layers.{}.mlp.gate_proj.weight"),
            w_down: get_tensor_vec!("model.layers.{}.mlp.down_proj.weight"),
            
            wq: get_tensor_vec!("model.layers.{}.self_attn.q_proj.weight"),
            wk: get_tensor_vec!("model.layers.{}.self_attn.k_proj.weight"),
            wv: get_tensor_vec!("model.layers.{}.self_attn.v_proj.weight"),
            wo: get_tensor_vec!("model.layers.{}.self_attn.o_proj.weight"),


        }
    }
}
