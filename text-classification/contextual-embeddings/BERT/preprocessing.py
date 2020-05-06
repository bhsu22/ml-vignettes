import numpy as np

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples


def chunk_token_sequences( inputs, mask_ids, segment_ids, labels, n_chunks, n_tokens, n_tokens_overlap):

    print(f"chunking input sequences...")
    print(f"Initial Input shape: {inputs.shape}")
    input_seq = inputs[:,1:-1]
    mask_seq = mask_ids[:,1:-1]
    seg_seq = segment_ids[:,1:-1]

    n,m = input_seq.shape
    input_chunks = []
    mask_chunks = []
    seg_chunks = []
    label_chunks = []

    for i, idx in zip(range(n_chunks), range(0, m, n_tokens)):
        start = max(0, idx-n_tokens_overlap-2) # -2 for cls/sep tokens
        end = start+n_tokens-2 # -2 for cls/sep tokens

        cls_tokens = np.array([101]*n).reshape(-1,1)
        sep_tokens = np.array([102 if i>0 else 0 for i in input_seq[:, end]]).reshape(-1,1)

        # if input sequence is padding, don't attend to it
        attend_cls= np.array([1 if i > 0 else 0 for i in input_seq[:, start]]).reshape(-1,1)
        attend_sep = np.array([1 if i>0 else 0 for i in input_seq[:, end]]).reshape(-1,1)

        input_chk = np.hstack((cls_tokens, input_seq[:, start:end], sep_tokens))
        mask_chk = np.hstack((attend_cls, mask_seq[:, start:end ], attend_sep))
        seg_chk = np.zeros(input_chk.shape) # only dealing with one segment

        input_chunks.append(input_chk)
        mask_chunks.append(mask_chk)
        seg_chunks.append(seg_chk)
        label_chunks.append(labels)

        print(f"=========chunk {i}=========")
        print(f"chunk shape: {input_chk.shape}")
        print(f"mask shape: {mask_chk.shape}")
        print(f"seg shap: {np.zeros(input_chk.shape).shape}")
        print(f"label shape: {labels.shape}")
    return input_chunks, mask_chunks, seg_chunks, label_chunks


def convert_text_to_example_chunks(texts, labels, n_chunks = 4, n_words=250, overlap_words=50):
    chunked_examples = []
    for text, label in zip(texts, labels):
        words = text.split(' ')

        chunks = [" ".join(words[max(0,idx-overlap_words): idx+n_words])
                  for idx in range(0, len(words), n_words)
                 ]

        while len(chunks) < n_chunks:
            # adds padding chunks for text that split into fewer than 4 chunks
            chunks.append(" ".join(["[PAD]"] * n_words))

        # every chunk gets the same label as the parent text
        chunks = convert_text_to_examples(chunks, [label]*n_chunks)
        chunked_examples.append(chunks)

    return np.array(chunked_examples)


def convert_example_chunks_to_features(example_chunks, tokenizer, max_seq_length, n_chunks):
    input_ids = []
    input_masks = []
    input_segs = []
    input_labels = []

    for i in range(n_chunks):
        # this can be run in parallel
        chunk = example_chunks[:,i]
        ids, masks, segs, labels = convert_examples_to_features(tokenizer, chunk, max_seq_length=max_seq_length)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segs.append(segs)
        input_labels.append(labels)
    return input_ids, input_masks, input_segs, input_labels
