"""Shared exact-mode verifier and immutable bank I/O for the checkpoint900 audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from source_snapshot.coin_change_final_verifier import extract_final_counts, verify_response, VERIFIER_NAME

ROOT = Path(__file__).resolve().parent


def extract_integer_list(response, key='counts'):
    if key != 'counts':
        raise ValueError('This frozen task accepts only the final literal counts field.')
    return extract_final_counts(response)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Refusing to replace {path}'
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write('\n')
    temporary.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read_jsonl(path) == rows, f'Refusing to replace {path}'
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as output:
        for row in rows:
            output.write(json.dumps(row, separators=(',', ':')) + '\n')
    temporary.replace(path)


def normalized_response(row, prompt):
    result = dict(row)
    verification = verify_response(row['response'], prompt)
    result['legacy_verification'] = {key: row.get(key) for key in ('reward', 'mode_key', 'error')}
    result.update(verification)
    result['verifier_name'] = VERIFIER_NAME
    result['final_counts'] = extract_final_counts(row['response'])
    result['format_valid'] = result['final_counts'] is not None
    result['evaluation_verifier_sha256'] = sha256(ROOT / 'source_snapshot' / 'coin_change_final_verifier.py')
    assert result['response_token_ids'] == row['response_token_ids']
    assert result['response_length'] == len(row['response_token_ids'])
    return result
