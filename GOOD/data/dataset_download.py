"""Install original GOOD archives from Hugging Face without changing their layout."""
import os
from pathlib import Path, PurePosixPath
import stat
from tempfile import TemporaryDirectory
from typing import Iterable, Union
from zipfile import ZipFile

from filelock import FileLock


DATASET_REPOSITORY = 'citrinegui/GOOD'
DATASET_REVISION = '29cf40c24604cefaf95f9bd8eb055608ad09da86'


def download_dataset(name: str, root: Union[str, Path], required_files: Iterable[Union[str, Path]]) -> None:
    """Ensure the loader's processed files exist, preserving any existing files.

    Complete local datasets need no network access. Otherwise the Hub manages
    resumable downloads and its shared archive cache; a per-dataset lock protects
    extraction by concurrent training jobs. A validated archive is extracted in
    a temporary directory before missing files are installed with atomic renames.
    Download, validation and filesystem errors propagate, so training cannot
    silently regenerate different splits after a failed download.
    """
    if not name.isidentifier():
        raise ValueError('Dataset name must be a single identifier')
    root = Path(os.path.abspath(Path(root).expanduser()))
    dataset_dir = root / name
    # Preserve the caller's layout here: complete legacy caches may use links.
    required = [Path(os.path.abspath(Path(path).expanduser())).relative_to(dataset_dir)
                for path in required_files]
    if not required:
        raise ValueError('At least one processed file is required')
    if _complete(dataset_dir, required):
        return

    # Resolve the configured root for writes, but not nested dataset links.
    root = root.resolve()
    dataset_dir = root / name
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ('.' + name + '.download.lock'))):
        if _complete(dataset_dir, required):
            return
        # Keep cached local training independent of Hub imports and credentials.
        from huggingface_hub import hf_hub_download

        archive_path = hf_hub_download(repo_id=DATASET_REPOSITORY, repo_type='dataset',
                                       filename=name + '.zip', revision=DATASET_REVISION)
        with TemporaryDirectory(prefix='.' + name + '.extract-', dir=str(root)) as staging:
            with ZipFile(archive_path) as archive:
                _validate_members(archive, name)
                archive.extractall(staging)
            extracted = Path(staging) / name
            if not _complete(extracted, required):
                raise ValueError('Dataset archive is missing required processed files for ' + name)
            _install_missing_files(extracted, dataset_dir)


def _complete(directory: Path, required: Iterable[Path]) -> bool:
    return all((directory / path).is_file() for path in required)


def _validate_members(archive: ZipFile, name: str) -> None:
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or not path.parts or path.parts[0] != name
                or '..' in path.parts or '\\' in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError('Unsafe path in dataset archive: ' + member.filename)


def _install_missing_files(extracted: Path, destination: Path) -> None:
    for source in extracted.rglob('*'):
        if not source.is_file():
            continue
        target = destination / source.relative_to(extracted)
        if target.is_file():
            continue
        # Only new writes need containment; existing linked files are untouched.
        target.resolve().relative_to(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(source), str(target))
