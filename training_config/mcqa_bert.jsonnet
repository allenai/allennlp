local train_size = 9741;
local batch_size = 8;
local gradient_accumulation_batch_size = 3;
local num_epochs = 4;
local learning_rate = 1e-5;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "bert-base-uncased";
local dataset_dir = "https://s3.amazonaws.com/commensenseqa/";
local cuda_device = 0;

{
  "dataset_reader": {
    "type": "transformer_mc_qa",
    "sample": -1,
    "num_choices": 5,
    "context_syntax": "q#a!",
    "add_prefix": {"q": "Q: ", "a": "A: "},
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  "datasets_for_vocab_creation": [],
  "train_data_path": dataset_dir + "train_rand_split.jsonl",
  "validation_data_path": dataset_dir + "dev_rand_split.jsonl",
  //"test_data_path": dataset_dir + "test_rand_split_no_answers.jsonl",
  //"evaluate_on_test": true,
  //"evaluate_custom": {
  //    "metadata_fields": "id,question_text,choice_text_list,correct_answer_index,answer_index,label_logits,label_probs"
  //},
  "model": {
    "type": "bert_mc_qa",
    // "transformer_weights_model": transformer_weights_model,
    "pretrained_model": transformer_model
  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size
  },
  "trainer": {
    "optimizer": {
        "type": "bert_adam",
        "lr": learning_rate,
        "warmup": warmup_ratio,
        "t_total": train_size / batch_size * num_epochs
    },
    "validation_metric": "+EM",
    "num_serialized_models_to_keep": 1,
    "should_log_learning_rate": true,
    "gradient_accumulation_steps": gradient_accumulation_batch_size,
    "num_epochs": num_epochs,
    "cuda_device": cuda_device
  }
}