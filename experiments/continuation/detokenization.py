"""Reproduce vLLM's fast incremental detokenization without changing any token.

The installed vLLM FastIncrementalDetokenizer uses tokenizers.DecodeStream,
primed with a safe prompt suffix. Incomplete UTF-8 bytes at a generation cap
can therefore have no emitted text, whereas batch decode emits U+FFFD. This
helper is only an exact replay check, never a repair of a model answer.
"""
from tokenizers.decoders import DecodeStream


def stream_decode(tokenizer, prompt_ids, response_ids):
    backend = tokenizer._tokenizer
    stream = DecodeStream(skip_special_tokens=True)
    suffix = prompt_ids
    if len(prompt_ids) > 4:
        for n in range(4, min(len(prompt_ids) + 1, 24)):
            candidate = prompt_ids[-n:]
            if '\ufffd' not in backend.decode(candidate):
                suffix = candidate
                break
    for token in suffix:
        stream.step(backend, int(token))
    return ''.join(stream.step(backend, int(token)) or '' for token in response_ids)


def verify_detokenization(tokenizer, prompt_ids, response_ids, recorded):
    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    if decoded == recorded:
        return 'batch_exact'
    if stream_decode(tokenizer, prompt_ids, response_ids) != recorded:
        raise ValueError('Neither batch nor exact incremental decoding matches recorded text')
    return 'incremental_exact'
