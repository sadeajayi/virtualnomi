import importlib


DATASET_UPDATE_MODULES = [
    "scripts.dataset_updates.add_adaora_phonetic_spelling",
    "scripts.dataset_updates.add_ebube_phonetic_spelling",
    "scripts.dataset_updates.deploy_discover_pronunciations",
    "scripts.dataset_updates.restore_audio_from_hf_history",
    "scripts.dataset_updates.update_ebube_ebubechukwu",
    "scripts.dataset_updates.update_morenikeji_meaning",
    "scripts.dataset_updates.update_yoruba_attributions",
    "scripts.dataset_updates.upload_audio_from_local_file",
]


def test_dataset_update_scripts_import_with_current_huggingface_hub(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    for module_name in DATASET_UPDATE_MODULES:
        importlib.import_module(module_name)
