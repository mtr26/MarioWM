from unittest.mock import Mock

import pytest

from world_model.conversion import ConversionConfig, convert_dataset, publish_cache


def test_publish_cache_creates_dataset_repo_and_uploads_validated_folder(
    synthetic_h5, tmp_path
):
    cache = convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=tmp_path / "cache",
            height=4,
            width=6,
            break_indices=(6,),
            workers=1,
        )
    )
    api = Mock()
    api.create_repo.return_value = "https://huggingface.co/datasets/user/mario"
    api.upload_folder.return_value = Mock(
        repo_url="https://huggingface.co/datasets/user/mario"
    )

    result = publish_cache(cache, "user/mario", private=True, api=api)

    api.create_repo.assert_called_once_with(
        repo_id="user/mario", repo_type="dataset", private=True, exist_ok=True
    )
    api.upload_folder.assert_called_once_with(
        folder_path=str(cache), repo_id="user/mario", repo_type="dataset"
    )
    assert result == "https://huggingface.co/datasets/user/mario"


def test_publish_cache_requires_namespace_and_name(synthetic_h5, tmp_path):
    cache = convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=tmp_path / "cache",
            height=4,
            width=6,
            break_indices=(6,),
            workers=1,
        )
    )

    with pytest.raises(ValueError, match="namespace/name"):
        publish_cache(cache, "missing-namespace", private=False, api=Mock())


def test_publish_cache_revalidates_before_network_call(synthetic_h5, tmp_path):
    cache = convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=tmp_path / "cache",
            height=4,
            width=6,
            break_indices=(6,),
            workers=1,
        )
    )
    actions = cache / "actions.npy"
    with actions.open("r+b") as stream:
        stream.seek(-1, 2)
        last_byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([last_byte[0] ^ 1]))
    api = Mock()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        publish_cache(cache, "user/mario", private=False, api=api)

    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()
