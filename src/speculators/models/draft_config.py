"""Load decoder configs with speculators-specific layer types."""

from transformers import AutoConfig, PretrainedConfig


def transformer_config_from_dict(value: dict) -> PretrainedConfig:
    """Accept Qwen3 ``mla`` layers without changing HF's global validation.

    HF validates its own layer vocabulary during construction. MLA uses the
    full-attention mask, so construct with that mask type, then restore the draft
    architecture labels for layer selection and checkpoint serialization.
    """
    values = dict(value)
    model_type = values.get("model_type", "qwen3")
    layer_types = values.get("layer_types")
    has_mla = layer_types is not None and "mla" in layer_types
    if has_mla:
        if model_type != "qwen3":
            raise ValueError("layer_types='mla' requires draft model_type='qwen3'")
        if not isinstance(layer_types, list):
            raise ValueError("layer_types must be a list of per-layer attention types")
        values["layer_types"] = [
            "full_attention" if kind == "mla" else kind for kind in layer_types
        ]
    default_config = AutoConfig.for_model(model_type)
    if has_mla and len(layer_types) != values.get(
        "num_hidden_layers", default_config.num_hidden_layers
    ):
        raise ValueError("layer_types length must equal num_hidden_layers")
    config_class = type(default_config)
    config = config_class.from_dict(values)
    if has_mla:
        config.layer_types = list(layer_types)
    return config
