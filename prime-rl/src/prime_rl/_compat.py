"""Compatibility fixes for dependency regressions."""

import transformers.modeling_flash_attention_utils as flash_attention_utils


if not hasattr(
    flash_attention_utils,
    "is_flash_attn_greater_or_equal_2_10",
):
    flash_attention_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True


try:
    import transformers.integrations.hub_kernels as hub_kernels
except ImportError:
    hub_kernels = None


if hub_kernels is not None:
    from huggingface_hub.errors import OfflineModeIsEnabled

    original_lazy_load_kernel = hub_kernels.lazy_load_kernel

    def patched_lazy_load_kernel(
        kernel_name,
        mapping=hub_kernels._KERNEL_MODULE_MAPPING,
    ):
        try:
            return original_lazy_load_kernel(kernel_name, mapping)
        except OfflineModeIsEnabled:
            mapping[kernel_name] = None
            return None

    hub_kernels.lazy_load_kernel = patched_lazy_load_kernel
