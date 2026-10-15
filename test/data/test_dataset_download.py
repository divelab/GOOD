"""Download behavior tests, runnable without importing the optional ML stack."""
import ast
import importlib.util
import os.path as osp
from pathlib import Path
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from zipfile import BadZipFile, ZipFile, ZipInfo

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'dataset_download', ROOT / 'GOOD/data/dataset_download.py')
downloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloads)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    calls = []
    archive_path = tmp_path / 'source.zip'
    module = ModuleType('huggingface_hub')

    def download(**kwargs):
        calls.append(kwargs)
        return str(archive_path)

    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', module)
    return calls, archive_path


def make_archive(path, entries):
    with ZipFile(path, 'w') as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def test_download_preserves_bytes_layout_and_cached_archive(tmp_path, hub):
    calls, source = hub
    root = tmp_path / 'datasets'
    expected = root / 'GOODCBAS/color/processed/no_shift.pt'
    make_archive(source, {'GOODCBAS/color/processed/no_shift.pt': b'original bytes'})
    downloads.download_dataset('GOODCBAS', root, [expected])
    assert expected.read_bytes() == b'original bytes'
    assert calls == [dict(repo_id='citrinegui/GOOD', repo_type='dataset',
                          filename='GOODCBAS.zip', revision=downloads.DATASET_REVISION)]
    assert source.is_file(), 'Never delete files from the shared Hugging Face cache'
    assert not list(root.glob('.GOODCBAS.extract-*'))
    downloads.download_dataset('GOODCBAS', root, [expected])
    assert len(calls) == 1, 'A complete dataset must not need the network'


def test_complete_legacy_cache_does_not_need_hub(tmp_path, hub):
    calls, _ = hub
    expected = tmp_path / 'GOODCBAS/color/processed/no_shift.pt'
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b'legacy')
    downloads.download_dataset('GOODCBAS', tmp_path, [expected])
    assert calls == []
    assert expected.read_bytes() == b'legacy'



@pytest.mark.parametrize('link_kind', ['dataset', 'processed', 'file'])
def test_complete_symlink_cache_reused_without_network(tmp_path, hub, link_kind):
    calls, _ = hub
    root = tmp_path / 'datasets'
    external = tmp_path / 'external'
    original = external / 'color/processed/no_shift.pt'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'legacy linked data')
    expected = root / 'GOODCBAS/color/processed/no_shift.pt'
    if link_kind == 'dataset':
        root.mkdir()
        (root / 'GOODCBAS').symlink_to(external, target_is_directory=True)
    elif link_kind == 'processed':
        expected.parent.parent.mkdir(parents=True)
        expected.parent.symlink_to(original.parent, target_is_directory=True)
    else:
        expected.parent.mkdir(parents=True)
        expected.symlink_to(original)
    downloads.download_dataset('GOODCBAS', root, [expected])
    assert calls == []
    assert original.read_bytes() == b'legacy linked data'
    assert not (root / '.GOODCBAS.download.lock').exists()


def test_partial_cache_preserves_linked_existing_file(tmp_path, hub):
    _, source = hub
    root = tmp_path / 'datasets'
    external = tmp_path / 'external.pt'
    external.write_bytes(b'linked original')
    existing = root / 'GOODCBAS/color/processed/no_shift.pt'
    existing.parent.mkdir(parents=True)
    existing.symlink_to(external)
    missing = existing.with_name('covariate.pt')
    make_archive(source, {'GOODCBAS/color/processed/no_shift.pt': b'archive data',
                          'GOODCBAS/color/processed/covariate.pt': b'new split'})
    downloads.download_dataset('GOODCBAS', root, [existing, missing])
    assert existing.is_symlink()
    assert external.read_bytes() == b'linked original'
    assert missing.read_bytes() == b'new split'


def test_incomplete_dataset_symlink_cannot_redirect_writes(tmp_path, hub):
    _, source = hub
    root = tmp_path / 'datasets'
    external = tmp_path / 'external'
    external.mkdir()
    root.mkdir()
    (root / 'GOODCBAS').symlink_to(external, target_is_directory=True)
    required = root / 'GOODCBAS/color/processed/no_shift.pt'
    make_archive(source, {'GOODCBAS/color/processed/no_shift.pt': b'new split'})
    with pytest.raises(ValueError):
        downloads.download_dataset('GOODCBAS', root, [required])
    assert list(external.iterdir()) == []


def test_symlinked_dataset_root_is_supported(tmp_path, hub):
    _, source = hub
    real_root = tmp_path / 'real-datasets'
    real_root.mkdir()
    root = tmp_path / 'linked-datasets'
    root.symlink_to(real_root, target_is_directory=True)
    required = root / 'GOODCBAS/color/processed/no_shift.pt'
    make_archive(source, {'GOODCBAS/color/processed/no_shift.pt': b'new split'})
    downloads.download_dataset('GOODCBAS', root, [required])
    assert required.read_bytes() == b'new split'


def test_partial_cache_repaired_without_changing_existing_files(tmp_path, hub):
    _, source = hub
    old = tmp_path / 'GOODCBAS/color/processed/no_shift.pt'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'locally generated')
    missing = old.with_name('covariate.pt')
    unrelated = tmp_path / 'GOODCBAS/custom/notes.txt'
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('keep')
    make_archive(source, {'GOODCBAS/color/processed/no_shift.pt': b'original',
                          'GOODCBAS/color/processed/covariate.pt': b'missing split'})
    downloads.download_dataset('GOODCBAS', tmp_path, [old, missing])
    assert old.read_bytes() == b'locally generated'
    assert missing.read_bytes() == b'missing split'
    assert unrelated.read_text() == 'keep'


def test_bad_archive_does_not_publish_and_retry_succeeds(tmp_path, hub):
    _, source = hub
    root = tmp_path / 'datasets'
    expected = root / 'GOODCBAS/color/processed/no_shift.pt'
    source.write_bytes(b'not a zip')
    with pytest.raises(BadZipFile):
        downloads.download_dataset('GOODCBAS', root, [expected])
    assert not (root / 'GOODCBAS').exists()
    assert not list(root.glob('.GOODCBAS.extract-*'))
    make_archive(source, {'GOODCBAS/color/processed/no_shift.pt': b'retry'})
    downloads.download_dataset('GOODCBAS', root, [expected])
    assert expected.read_bytes() == b'retry'


def test_missing_required_split_never_publishes_partial_data(tmp_path, hub):
    _, source = hub
    root = tmp_path / 'datasets'
    expected = root / 'GOODCBAS/color/processed/no_shift.pt'
    make_archive(source, {'GOODCBAS/color/processed/covariate.pt': b'wrong split'})
    with pytest.raises(ValueError, match='missing'):
        downloads.download_dataset('GOODCBAS', root, [expected])
    assert not (root / 'GOODCBAS').exists()
    assert not list(root.glob('.GOODCBAS.extract-*'))


@pytest.mark.parametrize('entry', ['../escape', '/absolute', 'GOODCBAS/../../escape',
                                  'OtherDataset/data.pt', 'GOODCBAS/../escape',
                                  'GOODCBAS\\escape'])
def test_unsafe_archive_never_writes_outside_stage(tmp_path, hub, entry):
    _, source = hub
    root = tmp_path / 'datasets'
    expected = root / 'GOODCBAS/color/processed/no_shift.pt'
    make_archive(source, {'GOODCBAS/color/processed/no_shift.pt': b'valid', entry: b'unsafe'})
    with pytest.raises(ValueError, match='archive'):
        downloads.download_dataset('GOODCBAS', root, [expected])
    assert not expected.exists()
    assert not (tmp_path / 'escape').exists()


def test_symlink_member_rejected(tmp_path, hub):
    _, source = hub
    info = ZipInfo('GOODCBAS/link')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(source, 'w') as archive:
        archive.writestr(info, '../outside')
    with pytest.raises(ValueError, match='archive'):
        downloads.download_dataset('GOODCBAS', tmp_path,
                                   [tmp_path / 'GOODCBAS/color/processed/no_shift.pt'])



def test_interrupted_install_cleans_staging_and_can_resume(tmp_path, hub, monkeypatch):
    _, source = hub
    required = [tmp_path / 'GOODCBAS/color/processed' / name
                for name in ['no_shift.pt', 'covariate.pt']]
    make_archive(source, {str(path.relative_to(tmp_path)): b'complete' for path in required})
    replace = downloads.os.replace
    calls = []

    def interrupted_replace(source_path, target_path):
        calls.append(target_path)
        if len(calls) == 2:
            raise OSError('disk unavailable')
        replace(source_path, target_path)

    with monkeypatch.context() as patch:
        patch.setattr(downloads.os, 'replace', interrupted_replace)
        with pytest.raises(OSError, match='disk unavailable'):
            downloads.download_dataset('GOODCBAS', tmp_path, required)
    assert sum(path.exists() for path in required) == 1
    assert not list(tmp_path.glob('.GOODCBAS.extract-*'))
    downloads.download_dataset('GOODCBAS', tmp_path, required)
    assert all(path.read_bytes() == b'complete' for path in required)


def test_existing_symlink_cannot_redirect_missing_files(tmp_path, hub):
    _, source = hub
    root = tmp_path / 'datasets'
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'GOODCBAS').mkdir(parents=True)
    (root / 'GOODCBAS/other').symlink_to(outside, target_is_directory=True)
    required = root / 'GOODCBAS/color/processed/no_shift.pt'
    make_archive(source, {'GOODCBAS/other/escape.pt': b'unsafe',
                          'GOODCBAS/color/processed/no_shift.pt': b'valid'})
    with pytest.raises(ValueError):
        downloads.download_dataset('GOODCBAS', root, [required])
    assert not (outside / 'escape.pt').exists()


def test_network_failure_is_visible_without_drive_fallback(tmp_path, hub, monkeypatch):
    def fail(**kwargs):
        raise ConnectionError('Hugging Face unavailable')
    monkeypatch.setattr(sys.modules['huggingface_hub'], 'hf_hub_download', fail)
    expected = tmp_path / 'GOODCBAS/color/processed/no_shift.pt'
    with pytest.raises(ConnectionError, match='Hugging Face unavailable'):
        downloads.download_dataset('GOODCBAS', tmp_path, [expected])
    assert not expected.exists()


def test_concurrent_downloads_publish_once(tmp_path, hub):
    calls, source = hub
    expected = tmp_path / 'GOODCBAS/color/processed/no_shift.pt'
    make_archive(source, {'GOODCBAS/color/processed/no_shift.pt': b'complete'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: downloads.download_dataset('GOODCBAS', tmp_path, [expected]), range(4)))
    assert len(calls) == 1
    assert expected.read_bytes() == b'complete'


LOADERS = {
    'good_arxiv': ('GOODArxiv', 'degree'),
    'good_cbas': ('GOODCBAS', 'color'),
    'good_cmnist': ('GOODCMNIST', 'color'),
    'good_cora': ('GOODCora', 'word'),
    'good_hiv': ('GOODHIV', 'scaffold'),
    'good_motif': ('GOODMotif', 'basis'),
    'good_pcba': ('GOODPCBA', 'scaffold'),
    'good_sst2': ('GOODSST2', 'length'),
    'good_twitch': ('GOODTwitch', 'language'),
    'good_twitter': ('GOODTwitter', 'length'),
    'good_webkb': ('GOODWebKB', 'university'),
    'good_zinc': ('GOODZINC', 'scaffold'),
    'lbap_core': ('LBAPcore', 'scaffold'),
}


def loader_without_ml(module_name, root):
    """Execute the actual download methods, omitting only heavyweight imports."""
    name, domain = LOADERS[module_name]
    source = ROOT / 'GOOD/data/good_datasets' / (module_name + '.py')
    parsed = ast.parse(source.read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.decorator_list = []
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
        '_download', 'download', 'raw_dir', 'processed_dir', 'processed_file_names'}]
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = dict(osp=osp, download_dataset=downloads.download_dataset)
    exec(compile(module, str(source), 'exec'), namespace)
    dataset_class = namespace[name]
    dataset_class.processed_paths = property(
        lambda self: [osp.join(self.processed_dir, name) for name in self.processed_file_names])
    dataset = dataset_class()
    dataset.root = str(root)
    dataset.name = name
    dataset.domain = domain
    dataset.generate = False
    return dataset


@pytest.mark.parametrize('module_name', LOADERS)
def test_each_loader_downloads_then_reuses_original_split_layout(tmp_path, hub, module_name):
    calls, source = hub
    root = tmp_path / 'datasets'
    dataset = loader_without_ml(module_name, root)
    entries = {str(Path(path).relative_to(root)): ('bytes-' + str(i)).encode()
               for i, path in enumerate(dataset.processed_paths)}
    make_archive(source, entries)
    dataset._download()
    assert len(calls) == 1
    assert calls[0]['filename'] == dataset.name + '.zip'
    for path, content in entries.items():
        assert (root / path).read_bytes() == content
    dataset._download()
    assert len(calls) == 1


@pytest.mark.parametrize('module_name', LOADERS)
def test_each_loader_generate_mode_skips_download(tmp_path, hub, module_name):
    calls, _ = hub
    root = tmp_path / 'datasets'
    dataset = loader_without_ml(module_name, root)
    dataset.generate = True
    dataset._download()
    assert calls == []
    assert not root.exists()
