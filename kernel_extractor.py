import os
import inspect
import hashlib
import functools
import logging
from pathlib import Path
from typing import Callable, Dict, Any

# Configure a logger for the toolkit
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [KernelExtractor] - %(levelname)s - %(message)s')

# Reference to the original triton.jit decorator, to be populated by the patch function
_original_triton_jit = None

class KernelExtractionManager:
    """
    A singleton class to manage the extraction of kernel code.
    Handles directory setup, file naming, and writing of kernel artifacts.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KernelExtractionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, output_dir: str = "vllm_kernels_output"):
        # Ensure __init__ is only run once for the singleton
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.output_dir = Path(output_dir)
        self.triton_dir = self.output_dir / "jit_triton"
        self.static_dir = self.output_dir / "static_cuda"
        self.manifest_path = self.output_dir / "manifest.json"
        
        self.triton_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest: Dict[str, Any] = {"triton_kernels": {}}
        self._initialized = True
        logger.info(f"Kernel extraction manager initialized. Output will be saved to: {self.output_dir.resolve()}")

    def save_triton_kernel(self, fn: Callable, compiled_kernel: Any):
        """
        Saves the source code and all compilation artifacts of a Triton kernel.
        """
        try:
            # Get the source code of the kernel function
            source_code = inspect.getsource(fn)
            
            # Create a unique name for the kernel based on its source code hash
            source_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]
            kernel_name = f"{fn.__name__}_{source_hash}"
            
            if kernel_name in self.manifest["triton_kernels"]:
                # This specific kernel version has already been saved
                return

            logger.info(f"Extracting Triton kernel: {fn.__name__} (hash: {source_hash})")
            
            kernel_path_prefix = self.triton_dir / kernel_name
            
            # 1. Save the Python source code
            py_path = kernel_path_prefix.with_suffix(".py")
            py_path.write_text(source_code)
            
            # 2. Save the compilation artifacts from the.asm dictionary
            artifacts = {}
            if hasattr(compiled_kernel, 'asm') and isinstance(compiled_kernel.asm, dict):
                for stage, code in compiled_kernel.asm.items():
                    if code:
                        artifact_path = kernel_path_prefix.with_suffix(f".{stage}")
                        artifact_path.write_text(code)
                        artifacts[stage] = str(artifact_path.resolve())

            # 3. Update the manifest
            self.manifest["triton_kernels"][kernel_name] = {
                "original_function": f"{fn.__module__}.{fn.__name__}",
                "python_source": str(py_path.resolve()),
                "artifacts": artifacts
            }
            self._write_manifest()

        except Exception as e:
            logger.error(f"Failed to save Triton kernel {fn.__name__}: {e}")

    def _write_manifest(self):
        """Writes the current manifest to disk."""
        import json
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=4)

    def copy_static_kernels(self, vllm_source_root: str):
        """
        Copies the static CUDA/C++ kernel sources from the vLLM csrc directory.
        """
        import shutil
        csrc_path = Path(vllm_source_root) / "vllm" / "csrc"
        if not csrc_path.exists():
            logger.warning(f"Could not find csrc directory at {csrc_path}. Skipping static kernel extraction.")
            return

        target_csrc_path = self.static_dir / "csrc"
        if target_csrc_path.exists():
            shutil.rmtree(target_csrc_path)
        
        shutil.copytree(csrc_path, target_csrc_path)
        logger.info(f"Copied static kernel sources from {csrc_path} to {target_csrc_path}")
        self.manifest["static_kernels_source_path"] = str(target_csrc_path.resolve())
        self._write_manifest()


def triton_jit_wrapper(fn: Callable) -> Callable:
    """
    A decorator that wraps the original triton.jit.
    It compiles the function and then hands it off to the manager for saving.
    """
    global _original_triton_jit
    if _original_triton_jit is None:
        raise RuntimeError("Original triton.jit not found. patch_triton_jit() must be called first.")

    # Apply the original triton.jit decorator to compile the kernel
    compiled_kernel = _original_triton_jit(fn)
    
    # Get the manager instance and save the artifacts
    manager = KernelExtractionManager()
    manager.save_triton_kernel(fn, compiled_kernel)
    
    # Return the compiled kernel so vLLM can use it
    return compiled_kernel


def patch_triton_jit():
    """
    Applies the monkey-patch to triton.jit.
    This function should be called once at the beginning of the application.
    """
    global _original_triton_jit
    try:
        import triton
        
        if hasattr(triton, 'jit') and not hasattr(triton.jit, '_is_extractor_patched'):
            logger.info("Found triton.jit. Applying interception patch.")
            _original_triton_jit = triton.jit
            
            # Replace the original decorator with our wrapper
            # We use functools.wraps to preserve the metadata of the original jit decorator
            @functools.wraps(_original_triton_jit)
            def patched_jit_decorator(*args, **kwargs):
                # The triton.jit decorator can be called in two ways:
                # 1. @triton.jit -> args is the function, kwargs is empty
                # 2. @triton.jit(debug=True) -> args is empty, kwargs has config
                if args and callable(args):
                    # Case 1: @triton.jit
                    fn = args
                    # Re-create the original decorator call for this specific function
                    _original_triton_jit_fn = _original_triton_jit
                    compiled_kernel = _original_triton_jit_fn(fn)
                    manager = KernelExtractionManager()
                    manager.save_triton_kernel(fn, compiled_kernel)
                    return compiled_kernel
                else:
                    # Case 2: @triton.jit(...)
                    # We need to return a decorator that will be applied to the function
                    def decorator(fn: Callable):
                        # Re-create the original decorator with its arguments
                        _original_triton_jit_configured = _original_triton_jit(*args, **kwargs)
                        compiled_kernel = _original_triton_jit_configured(fn)
                        manager = KernelExtractionManager()
                        manager.save_triton_kernel(fn, compiled_kernel)
                        return compiled_kernel
                    return decorator

            triton.jit = patched_jit_decorator
            triton.jit._is_extractor_patched = True
            logger.info("Successfully patched triton.jit for kernel extraction.")
        elif hasattr(triton.jit, '_is_extractor_patched'):
            logger.info("triton.jit is already patched. Skipping.")
        else:
            logger.error("Could not find triton.jit to patch.")
            
    except ImportError:
        logger.warning("Triton library not found. Skipping patch for Triton kernels.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while patching triton.jit: {e}")