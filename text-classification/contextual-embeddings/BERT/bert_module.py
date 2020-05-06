"""Create a BERT tf.hub Module from a checkpoint file.

Creates an easier to use BERT module from a checkpoint. 
Needed to create a Keras Layer for BERT.

Ref:
    https://github.com/gaphex/bert_experimental/blob/master/bert_experimental/finetuning/modeling.py
    https://towardsdatascience.com/fine-tuning-bert-with-keras-and-tf-module-ed24ea91cff2
"""
import tensorflow as tf
import tensorflow_hub as hub

from bert import modeling

def build_module_fn(config_path, vocab_path, do_lower_case=True, seq_layer=-1, tok_layer=-7):

    def bert_module_fn(is_training):
        """Spec function for a token embedding module."""

        input_ids = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
        input_mask = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
        token_type = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")

        config = modeling.BertConfig.from_json_file(config_path)
        model = modeling.BertModel(config=config, is_training=is_training,
                          input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type)
          
        seq_output = model.all_encoder_layers[seq_layer]
        tok_output = model.all_encoder_layers[tok_layer]
        pool_output = model.get_pooled_output()

        config_file = tf.constant(value=config_path, dtype=tf.string, name="config_file")
        vocab_file = tf.constant(value=vocab_path, dtype=tf.string, name="vocab_file")
        lower_case = tf.constant(do_lower_case)

        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS, config_file)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS, vocab_file)
        
        input_map = {"input_ids": input_ids,
                     "input_mask": input_mask,
                     "segment_ids": token_type}
        
        output_map = {"pooled_output": pool_output,
                      "sequence_output": seq_output,
                      "token_output": tok_output}

        output_info_map = {"vocab_file": vocab_file,
                           "do_lower_case": lower_case}
                
        hub.add_signature(name="tokens", inputs=input_map, outputs=output_map)
        hub.add_signature(name="tokenization_info", inputs={}, outputs=output_info_map)

    return bert_module_fn

def build_bert_module(config_path, vocab_path, ckpt_path, out_path, 
                      seq_layer=-1, tok_layer=-7):

    tags_and_args = []
    for is_training in (True, False):
        tags = set()
        if is_training:
            tags.add("train")
        tags_and_args.append((tags, dict(is_training=is_training)))

    module_fn = build_module_fn(config_path, vocab_path, seq_layer=seq_layer, tok_layer=tok_layer)
    spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args)
    try:
        spec.export(out_path, checkpoint_path=ckpt_path)
    except tf.errors.AlreadyExistsError:
        print(f"BERT module in {out_path} already exists.")