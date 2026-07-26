"""Adapter Manager for Galatea.

This module provides an abstraction for managing LoRA adapters (PEFT) in a
language model. It supports multiple simultaneous adapters, dynamic loading,
gradient updates, and entropy computation with temporary adapter switching.

Key responsibilities:
- Load/unload adapters by name
- Activate one or more adapters (weight summation)
- Retrieve adapter parameters for saving/restoring
- Apply gradient steps to specific adapters
- Compute model entropy with a temporarily activated adapter
"""

from abc import ABC, abstractmethod
from typing import Literal, cast

import torch

from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model
from torch import nn
from transformers import PreTrainedModel

from ..utils.helpers import entropy
from .adapter_state_store import AdapterStateStore


class AdapterManager(ABC):
    """Abstract interface for managing LoRA adapters in a PEFT-wrapped model.

    This interface defines all operations needed for Galatea's adaptive
    learning: dynamic adapter creation, multi-adapter activation,
    and gradient application.

    Design decisions:
    - Adapters are identified by string names
        (typically user IDs or combination keys).
    - Multiple adapters can be active simultaneously; their weights are summed
        during forward pass.
    - Adapter parameters can be stored persistently via AdapterStateStore.
    - Entropy computation can temporarily enable an additional adapter
        for threat evaluation.
    """

    @abstractmethod
    def load_adapter(
        self,
        name: str,
        weights: dict[str, torch.Tensor] | None = None
    ) -> None:
        """Load an adapter with the given name.

        If the adapter already exists, this method does nothing.

        Important:
            This method does NOT activate the adapter; you must call
            `set_active_adapters()` or `set_active_adapter()`
            separately to use it.

        Args:
            name: Unique identifier for the adapter
                (e.g., user_id or "user_123+knowledge").
            weights: Optional pre-trained weights to initialise the adapter.
                If not provided, the adapter is created with random
                initialisation (PEFT's default behaviour for newly
                added adapters).

        Notes:
            - The adapter is created with the same LoRA configuration as
                defined in the constructor.
            - After loading, the adapter's parameters are stored in the
                internal state store.

        """

    @abstractmethod
    def unload_adapter(self, name: str) -> None:
        """Unload (remove) an adapter from the model.

        If the adapter is currently active, it is deactivated first.

        Args:
            name: Name of the adapter to unload.

        Raises:
            ValueError: If no adapter with the given name is found.

        Effects:
            - The adapter is removed from the model's configuration.
            - The adapter parameters are deleted from the state store.
            - If the adapter was in the active list, it is removed
                and the remaining adapters are re-activated; if no
                adapters remain, all adapters are disabled.

        """

    @abstractmethod
    def set_active_adapters(self, names: list[str]) -> None:
        """Activate one or more adapters simultaneously.

        When multiple adapters are active, their weights are
        combined (summed) during the forward pass. This allows combining,
        for example, a user-specific adapter with a domain knowledge adapter.

        Important:
            All given adapters must already be loaded (via load_adapter).
            If any adapter is not loaded, this method will raise an error.

        Args:
            names: List of adapter names to activate.

        Raises:
            ValueError: If any adapter name is not loaded.

        Effects:
            - The internal active adapter list is updated.
            - The underlying PEFT model's active adapters are set accordingly.
            - If the list is empty, all adapters are disabled
                (base model only).

        """

    @abstractmethod
    def get_active_adapters(self) -> list[str]:
        """Return the list of currently active adapters.

        Returns:
            A copy of the internal list of active adapter names.
            The order may not be significant.

        """

    @abstractmethod
    def get_adapter_parameters(self, name: str) -> dict[str, torch.Tensor]:
        """Retrieve the parameters of a specific adapter.

        This method is used for state saving, restoration, and inspection.

        Args:
            name: Name of the adapter.

        Returns:
            A dictionary mapping parameter names to torch.Tensor objects.

        Raises:
            ValueError: If the adapter is not loaded.

        """

    @abstractmethod
    def apply_gradient_step(
        self,
        name: str,
        gradient_step: dict[str, torch.Tensor],
        learning_rate: float,
    ) -> None:
        """Apply an SGD update step to the parameters of a specific adapter.

        This method modifies only the parameters of the specified adapter.
        It is used during online learning to adjust the adapter weights.

        Args:
            name: Name of the target adapter.
            gradient_step: Dictionary mapping parameter names
                to gradient tensors.
            learning_rate: The learning rate (step size) for the update.

        Notes:
            - The update is performed in-place using torch.Tensor.sub_()
                to avoid creating new tensors and to respect the no_grad
                context.
            - Only parameters that appear in the gradient_step
                dictionary are updated.
            - Parameters that belong to other adapters are left untouched.

        """

    @abstractmethod
    def compute_entropy_with_adapter(
        self,
        input_ids: torch.Tensor,
        adapter_name: str,
    ) -> float:
        """Temporarily activate the specified adapter (in addition to the
        currently active ones) and compute the entropy
        of the model's logits.

        This method is used by the ThreatPrism to evaluate how a potential
        gradient update would affect the model's uncertainty (entropy),
        which is a proxy for instability or threat.

        The method:
        1. Saves the current active adapters.
        2. Temporarily adds the given adapter (loading it if necessary).
        3. Computes the forward pass with the temporary adapter combination.
        4. Restores the original active adapters.
        5. Returns the entropy of the logits.

        Args:
            input_ids: Input token IDs to feed into the model.
            adapter_name: Name of the adapter to temporarily add.

        Returns:
            The entropy (float) of the model's output logits distribution.

        """

    @abstractmethod
    def list_adapters(self) -> list[str]:
        """Return a list of all loaded adapter names.

        Returns:
            List of names of adapters currently present
            in the model's configuration.

        """

    # Convenience wrappers (non-abstract) for single-adapter operations.

    def set_active_adapter(self, name: str) -> None:
        """Set exactly one adapter active. This is a convenience wrapper around
        set_active_adapters([name]).

        Args:
            name: Adapter name to activate.

        """
        self.set_active_adapters([name])

    def get_active_adapter(self) -> str | None:
        """Return the first active adapter, or None if no adapters are active.

        This is a convenience method when only one adapter is expected.

        Returns:
            The first active adapter name, or None.

        """
        adapters = self.get_active_adapters()
        return adapters[0] if adapters else None


class HuggingFaceAdapterManager(AdapterManager):
    """Implementation of AdapterManager for HuggingFace models
    with PEFT (LoRA).

    This class manages PEFT adapters on top of a base PreTrainedModel.
    It supports:
        - Dynamic creation of new adapters with consistent LoRA configuration.
        - Activation of multiple adapters simultaneously (weight summation).
        - In-place gradient updates for specific adapters.
        - Temporary adapter switching for entropy computation.
        - Integration with an external AdapterStateStore for
            persisting adapter weights.

    The manager stores the underlying PEFT model (`_peft_model`), the active
    adapter list, and a state store for parameter snapshots. It also keeps the
    LoRA configuration parameters (r, alpha, target modules, etc.) that will
    be used when creating new adapters.

    Implementation notes:
        - The `_get_peft()` method ensures that the PEFT model is initialised
            and returns it; it raises a RuntimeError if the model is None.
        - The `load_adapter` method creates a new adapter with the
            stored LoRA config.
        - The `set_active_adapters` method updates both the internal list and
            the PEFT model.
        - The `apply_gradient_step` method uses `param.sub_()` for
            in-place updates.
        - The `compute_entropy_with_adapter` method temporarily modifies
            the active adapters and restores them afterwards.
    """

    def __init__(
        self,
        base_model: nn.Module,
        state_store: AdapterStateStore | None = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        target_modules: list[str] | None = None,
        lora_dropout: float = 0.0,
        bias: Literal['none', 'all', 'lora_only'] = 'none',
        task_type: str = 'CAUSAL_LM',
    ):
        """Initialise the HuggingFaceAdapterManager.

        Args:
            base_model: The base model (could be a PreTrainedModel or
                already a PeftModel).
            state_store: Optional external state store for adapter weights.
                If not provided, an in-memory store is created.
            lora_r: LoRA rank (dimension of low-rank matrices).
            lora_alpha: LoRA alpha scaling parameter.
            target_modules: List of module names to apply
                LoRA to (e.g., ["q_proj", "v_proj"]).
            lora_dropout: Dropout rate for LoRA layers.
            bias: Bias handling strategy ("none", "all", "lora_only").
            task_type: Task type (e.g., "CAUSAL_LM" for
                causal language modelling).

        Raises:
            TypeError: If base_model is not a PreTrainedModel and
                not already a PeftModel.

        """
        self._state_store = state_store or AdapterStateStore()

        # Store LoRA configuration for later use when creating new adapters.
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._target_modules = target_modules or ['q_proj', 'v_proj']
        self._lora_dropout = lora_dropout
        self._bias: Literal['none', 'all', 'lora_only'] = bias
        self._task_type = task_type

        # Initialise or wrap the base model as a PEFT model.
        # If it's already a PeftModel or PeftMixedModel, just reference it.
        if isinstance(base_model, (PeftModel, PeftMixedModel)):
            self._peft_model: PeftModel | PeftMixedModel | None = base_model
        else:
            # Ensure it's a PreTrainedModel,
            # because get_peft_model requires that.
            if not isinstance(base_model, PreTrainedModel):
                raise TypeError(
                    'base_model must be a PreTrainedModel, '
                    f'got {type(base_model)}'
                )
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules or ['q_proj', 'v_proj'],
                lora_dropout=lora_dropout,
                bias=bias,
                task_type=task_type,
            )
            peft_model = get_peft_model(base_model, config)
            # Disable all adapters initially (we want
            # base model only by default).
            peft_model.disable_adapter()
            # Remove the default adapter that get_peft_model creates.
            if 'default' in peft_model.peft_config:
                del peft_model.peft_config['default']
            self._peft_model = peft_model

        # Internal list of currently active adapters (by name).
        self._active_adapters: list[str] = []

    def _get_peft(self) -> PeftModel | PeftMixedModel:
        """Return the underlying PEFT model, ensuring it is not None.

        This method is used internally by all methods that need to access
        the model. It raises a RuntimeError if the model is not initialised.

        Returns:
            The PEFT model (PeftModel or PeftMixedModel).

        Raises:
            RuntimeError: If the PEFT model is not initialised.

        """
        if self._peft_model is None:
            raise RuntimeError('PEFT model is not initialized.')
        return self._peft_model

    def load_adapter(
        self,
        name: str,
        weights: dict[str, torch.Tensor] | None = None
    ) -> None:
        """Load (create) a new adapter with the given name.

        If the adapter already exists, nothing is done. Otherwise, a new
        adapter is created using the LoRA configuration stored
        at initialisation.

        After creation, if weights are provided, they are loaded into the
        adapter. Then, the adapter's parameters are saved into the
        state store.

        Args:
            name: Name of the adapter.
            weights: Optional initial weights for the adapter.

        """
        peft = self._get_peft()
        # Check if already loaded; if so, skip.
        if name in peft.peft_config:
            return

        # Create a new LoraConfig based on the stored parameters.
        config = LoraConfig(
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            target_modules=self._target_modules,
            lora_dropout=self._lora_dropout,
            bias=self._bias,
            task_type=self._task_type,
        )
        # Add the adapter to the model.
        peft.add_adapter(name, config)

        # If weights are given, load them into the adapter parameters.
        if weights is not None:
            self._load_adapter_weights(name, weights)

        # Store a snapshot of the adapter's parameters in the state store.
        self._state_store.save(name, self.get_adapter_parameters(name))

    def _load_adapter_weights(
        self,
        name: str,
        weights: dict[str, torch.Tensor]
    ) -> None:
        """Load weights into a specific adapter by name.

        This method iterates over all parameters of the PEFT model and copies
        the corresponding values from the provided weights dictionary.

        Args:
            name: Name of the adapter (not currently used for
                lookup, but kept for symmetry).
            weights: Dictionary of parameter names to tensor values.

        """
        peft = self._get_peft()
        for param_name, param in peft.named_parameters():
            if param_name in weights:
                param.data.copy_(weights[param_name])

    def unload_adapter(self, name: str) -> None:
        """Unload (remove) an adapter from the model.

        If the adapter is active, it is deactivated first
        (removed from the active list). Then the adapter is
        removed from the model's configuration, and its
        parameters are deleted from the state store.

        Args:
            name: Name of the adapter.

        Raises:
            ValueError: If the adapter is not found.

        """
        peft = self._get_peft()
        if name not in peft.peft_config:
            raise ValueError(f"Adapter '{name}' not found.")

        # Remove the adapter from the model configuration.
        del peft.peft_config[name]

        # Delete its parameters from the state store.
        self._state_store.delete(name)

        # If the adapter was active, remove it from the active list.
        if name in self._active_adapters:
            self._active_adapters.remove(name)
            # Re-activate the remaining adapters or disable all.
            if self._active_adapters:
                # PEFT supports a list of adapters, but mypy's
                # type hint expects str.
                # We suppress the type error because it's safe.
                peft.set_adapter(
                    self._active_adapters  # type: ignore[arg-type]
                )
            else:
                peft.disable_adapter()

    def set_active_adapters(self, names: list[str]) -> None:
        """Activate one or more adapters. All adapters must already be loaded;
        if any are missing, they are loaded automatically via load_adapter.

        Args:
            names: List of adapter names.

        """
        peft = self._get_peft()
        # Ensure all adapters are loaded (load if missing).
        for name in names:
            if name not in peft.peft_config:
                self.load_adapter(name)

        # Update the internal list and the model's active adapters.
        self._active_adapters = names.copy()
        if names:
            # PEFT accepts a list, but type hint says str.
            peft.set_adapter(names)  # type: ignore[arg-type]
        else:
            peft.disable_adapter()

    def get_active_adapters(self) -> list[str]:
        """Return a copy of the active adapters list."""
        return self._active_adapters.copy()

    def get_adapter_parameters(self, name: str) -> dict[str, torch.Tensor]:
        """Retrieve the parameters of a specific adapter.

        Args:
            name: Adapter name.

        Returns:
            Dictionary of parameter names to tensors.

        Raises:
            ValueError: If adapter not found
                (handled implicitly by empty dict?).

        """
        peft = self._get_peft()
        params: dict[str, torch.Tensor] = {}
        # Parameter names follow the pattern:
        # ... .lora_A.<name>.* or .lora_B.<name>.*
        for param_name, param in peft.named_parameters():
            if (
                f'.lora_A.{name}.' in param_name
                or f'.lora_B.{name}.' in param_name
            ):
                # cast to Tensor because Parameter is a subclass of Tensor.
                params[param_name] = cast(torch.Tensor, param)
        return params

    def apply_gradient_step(
        self,
        name: str,
        gradient_step: dict[str, torch.Tensor],
        learning_rate: float,
    ) -> None:
        """Apply an SGD update step to the parameters of a specific adapter.

        For each parameter that appears in the gradient_step dictionary
        and belongs to the specified adapter
        (determined by the .lora_A.<name>. pattern),
        subtract learning_rate * gradient from the parameter.

        This is done in-place inside a torch.no_grad() context to
        avoid tracking gradients for this update.

        Args:
            name: Target adapter name.
            gradient_step: Dict of parameter gradients.
            learning_rate: Step size.

        """
        peft = self._get_peft()
        with torch.no_grad():
            for param_name, param in peft.named_parameters():
                if param_name in gradient_step and gradient_step[param_name] is not None:  # noqa: E501
                    if f'.lora_A.{name}.' in param_name or f'.lora_B.{name}.' in param_name:  # noqa: E501
                        # Use in-place subtraction to avoid
                        # creating new tensors.
                        param.sub_(learning_rate * gradient_step[param_name])

    def compute_entropy_with_adapter(
        self,
        input_ids: torch.Tensor,
        adapter_name: str,
    ) -> float:
        """Compute the entropy of the model's logits while temporarily
        adding the specified adapter to the current active set.

        This method is used by ThreatPrism to evaluate the effect
        of a potential gradient update on model uncertainty.

        The temporary activation is done by:
            1. Saving the current active adapters.
            2. Ensuring the target adapter is loaded.
            3. Adding it to the active set (if not already active).
            4. Running a forward pass to obtain logits.
            5. Computing entropy.
            6. Restoring the original active adapters.

        Args:
            input_ids: Input token IDs.
            adapter_name: Name of the adapter to temporarily add.

        Returns:
            The entropy of the logits (float).

        """
        peft = self._get_peft()
        current_adapters = self._active_adapters.copy()

        # Ensure the adapter is loaded (create if missing).
        if adapter_name not in peft.peft_config:
            self.load_adapter(adapter_name)

        # Temporarily add the adapter to the active set.
        if adapter_name not in current_adapters:
            temp_adapters = current_adapters + [adapter_name]
            peft.set_adapter(temp_adapters)  # type: ignore[arg-type]

        # Forward pass and entropy computation.
        with torch.no_grad():
            logits = peft(input_ids).logits

        # Restore the original active adapters.
        if current_adapters:
            peft.set_adapter(current_adapters)  # type: ignore[arg-type]
        else:
            peft.disable_adapter()

        return entropy(logits)

    def list_adapters(self) -> list[str]:
        """Return a list of all loaded adapter names.

        Returns:
            List of adapter names from the model's configuration.

        """
        peft = self._get_peft()
        return list(peft.peft_config.keys())

    def save_adapter(self, name: str, path: str) -> None:
        """Save the specified adapter to disk using PEFT's save_pretrained.

        Args:
            name: Adapter name.
            path: Directory path to save the adapter files.

        Raises:
            ValueError: If the adapter does not exist.

        """
        peft = self._get_peft()
        if name not in peft.peft_config:
            raise ValueError(f"Adapter '{name}' not found.")
        peft.save_pretrained(path, adapter_name=name)

    def load_adapter_from_disk(self, name: str, path: str) -> None:
        """Load an adapter from a previously saved directory.

        If the adapter is already loaded, this method does nothing.

        Args:
            name: Adapter name to load.
            path: Path to the adapter directory.

        """
        peft = self._get_peft()
        if name in peft.peft_config:
            return  # Already loaded.
        peft.load_adapter(path, adapter_name=name)
        # Save the loaded parameters into the state store for quick access.
        self._state_store.save(name, self.get_adapter_parameters(name))
