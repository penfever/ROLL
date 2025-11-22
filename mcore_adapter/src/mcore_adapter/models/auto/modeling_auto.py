import json
import os
from collections import OrderedDict

from transformers import AutoConfig as HfAutoConfig
from transformers.configuration_utils import CONFIG_NAME as HF_CONFIG_NAME

from ...constants import MCA_CONFIG_NAME
from ...utils import get_logger
from ..model_factory import McaGPTModel, VirtualModels


logger = get_logger(__name__)


MODEL_MAPPING = OrderedDict()

MODEL_TYPE_ALIASES = {
    "qwen3-coder": "qwen3_moe",
    "qwen3_coder": "qwen3_moe",
}


def register_model(model_type, cls=None):
    def decorator(cls):
        if model_type in MODEL_MAPPING:
            logger.warning(f"Model for model type {model_type} already registered, overriding!")
        MODEL_MAPPING[model_type] = cls
        return cls

    if cls is not None:
        return decorator(cls)
    return decorator


def get_model_cls(model_type) -> "McaGPTModel":
    cls = MODEL_MAPPING.get(model_type)
    if cls is None:
        logger.warning(f"No model found for model type {model_type}, use McaGPTModel!")
        cls = McaGPTModel
    return cls


class AutoModel:
    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        config_file = os.path.join(model_name_or_path, MCA_CONFIG_NAME)
        model_type = None
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            config_values = json.loads(text)
            model_type = config_values.get("hf_model_type")
        else:
            hf_config_path = os.path.join(model_name_or_path, HF_CONFIG_NAME)
            if os.path.isfile(hf_config_path):
                logger.info(f"Did not find {config_file}, loading HuggingFace config from {model_name_or_path}")
                hf_config = HfAutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                model_type = hf_config.model_type
            else:
                try:
                    logger.info(
                        f"Downloading HuggingFace config for {model_name_or_path} via AutoConfig.from_pretrained."
                    )
                    hf_config = HfAutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                    model_type = hf_config.model_type
                except Exception as exc:
                    lowered_name = str(model_name_or_path).lower()
                    for key, alias in MODEL_TYPE_ALIASES.items():
                        if key in lowered_name:
                            logger.warning(
                                f"Failed to load config for {model_name_or_path}; using alias '{alias}'."
                            )
                            model_type = alias
                            break
                    if model_type is None:
                        raise ValueError(f"No valid config found in {model_name_or_path}") from exc

        if model_type is None:
            lowered_name = str(model_name_or_path).lower()
            for key, alias in MODEL_TYPE_ALIASES.items():
                if key in lowered_name:
                    logger.warning(
                        f"Could not infer model type from config for {model_name_or_path}, "
                        f"falling back to alias '{alias}'."
                    )
                    model_type = alias
                    break

        elif model_type in MODEL_TYPE_ALIASES and model_type not in MODEL_MAPPING:
            alias = MODEL_TYPE_ALIASES[model_type]
            logger.warning(
                f"Model type '{model_type}' not registered, using alias '{alias}' for {model_name_or_path}."
            )
            model_type = alias

        if model_type is None:
            raise ValueError(f"No valid config found in {model_name_or_path}")
        model_class = get_model_cls(model_type)
        return model_class.from_pretrained(model_name_or_path, *args, **kwargs)

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        model_type = config.hf_model_type
        model_class = get_model_cls(model_type)
        return VirtualModels(model_class, config=config, *args, **kwargs)
