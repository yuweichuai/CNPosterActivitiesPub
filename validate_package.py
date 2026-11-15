"""Validate supplied files, pseudonymous keys and analysis panel structure (no Stata)."""
import ast
import hashlib
import json
from pathlib import Path
import re

import nbformat
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    source = pd.read_parquet(ROOT / 'data/source_post_features_processed.parquet')
    assert source['tweet_id'].is_unique
    assert source[['tweet_id', 'user_id']].notna().all().all()
    assert source['treated'].isin([0, 1]).all()
    event_ids = set(source['tweet_id'])
    deleted_ids = set(pd.read_parquet(ROOT / 'data/deleted_posts_daily.parquet')['tweet_id'])
    permitted_text = {'tweet_id', 'user_id', 'displayed', 'original_group', 'first_correction_quarter'}
    prohibited = {'username', 'screen_name', 'tweet_text', 'text', 'user_tweet_text',
                  'noteId', 'noteAuthorParticipantId', 'reply_id', 'conversation_id',
                  'entities.urls', 'entities.user_mentions', 'count_start_time', 'count_end_time'}
    for item in manifest:
        path = ROOT / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
        data = pd.read_parquet(path)
        assert len(data) == item['rows'] and list(data.columns) == item['columns'], item['path']
        assert not set(data.columns) & prohibited, item['path']
        for col in data:
            dtype = data[col].dtype
            assert not pd.api.types.is_datetime64_any_dtype(dtype), (item['path'], col)
            if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
                assert col in permitted_text, (item['path'], col)
            if col in ('tweet_id', 'user_id'):
                prefix = 'E' if col == 'tweet_id' else 'U'
                assert data[col].notna().all()
                assert data[col].str.fullmatch(prefix + r'\d{7}').all(), (item['path'], col)
        time = next((c for c in ('days_from_display', 'weeks_from_display') if c in data), None)
        if time:
            assert not data.duplicated(['tweet_id', time]).any(), item['path']
            if path.name.startswith(('ts_posts_', 'deleted_posts_')):
                extent = 28 if time == 'days_from_display' else 4
                assert data[time].isin(list(range(-extent, 0)) + list(range(1, extent + 1))).all()
            assert set(data['tweet_id']) <= event_ids | deleted_ids, item['path']
        if 'tweet_count' in data:
            assert data['tweet_count'].ge(0).all()
        for col in data:
            if col.startswith('tweet_count_'):
                assert data[col].dropna().ge(0).all()
    for path in ROOT.glob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                assert not cell.outputs and cell.execution_count is None, path.name
                ast.parse(cell.source)
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text())
    content = json.loads((ROOT / 'data/content_counts/manifest.json').read_text())
    assert len(content) == len({(x['source_alias'], x['variable'], x['time_column']) for x in content})
    for item in content:
        assert (ROOT / item['path']).is_file()
    print(f'PASS: {len(manifest)} datasets; {len(list(ROOT.glob("*.ipynb")))} clean notebooks; '
          f'{len(content)} content-count tables; consistent pseudonymous keys and valid panels.')


if __name__ == '__main__':
    main()
