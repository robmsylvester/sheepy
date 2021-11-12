from lib_ml_framework.src.common.logger import get_std_out_logger
import numpy as np
import onnx
import onnxruntime
import os.path
import pathlib
from torch.nn.modules import Module
import torch.onnx


def convert_model_to_onnx_format(model: Module, model_input: torch.Tensor, output_file_name: str,
                                 onnx_models_path: str = None, opset_version: int = 11):
    """
    This method takes a PyTorch Lightning model and converts it to the ONNX format. The result is a .onnx file placed in
    the "onnx_models" folder at the root of the project.

    Quoting an ONNX tutorial (https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb) :
        "The ONNX exporter is a trace-based exporter, which means that it operates by executing your model once, and
        exporting the operators which were actually run during this run. This means that if your model is dynamic, e.g.,
        changes behavior depending on input data, the export won’t be accurate.

        Similarly, a trace is might be valid only for a specific input size (which is one reason why we require explicit
        inputs on tracing). Most of the operators export size-agnostic versions and should work on different batch sizes
        or input sizes. We recommend examining the model trace and making sure the traced operators look reasonable."

    Arguments:
        model (torch.nn.Module): Model to export to ONNX format
        model_input (tuple): Since the converter is trace-based, it needs to execute a forward pass with some data. It
            can be random values but needs to have the right structure.
        output_file_name (str): The name of the model
        onnx_models_path (str): The path of the the onnx_models folder. If None, it will be retrieved automatically.
        opset_version (int): The version of the operator set to use. See
            https://github.com/microsoft/onnxjs/blob/master/docs/operators.md for the list of supported operators by
            ONNX.js.
    """
    if onnx_models_path is None:
        onnx_models_path = get_onnx_models_path()
    # Create onnx_models folder if missing
    if not os.path.exists(onnx_models_path):
        os.mkdir(onnx_models_path)
    output_path = onnx_models_path + output_file_name + ".onnx"
    model.eval()  # set the model to inference mode
    torch.onnx.export(model=model, args=model_input, f=output_path, do_constant_folding=True,
                      opset_version=opset_version)
    logger = get_std_out_logger()
    logger.info(f"Torch model {output_file_name} has been converted to the ONNX format and saved to {output_path}")


def validate_onnx_model(model_name: str, onnx_models_path: str = None, torch_model: Module = None,
                        model_input: torch.Tensor = None):
    """
    This function validates the structure of the ONNX model created with the convert_model_to_onnx_format method. If the
    torch_model and model_input arguments are provided, it will also compare the results of the two models to determine
    if they are close enough.

    Warning - A workaround is included in this method to prevent an error caused by the double import of libomp.dylib. I
        do not know where those libraries are include and I could not find a cleaner solution. The workaround is given
        by the error message:
            "OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is
            dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure
            that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP
            runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment
            variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes
            or silently produce incorrect results. For more information, please see http://openmp.llvm.org/"

    Arguments:
        model_name (str): The name of the model to search for in the onnx_models folder.
        onnx_models_path (str): The path of the the onnx_models folder. If None, it will be retrieved automatically.
        torch_model (Module): The Pytorch version of the ONNX model (the one used for conversion).
        model_input (torch.Tensor): The input used for inference with both models to compare outputs.
    """
    # Add onnx suffix if missing
    if not str.endswith(model_name, ".onnx"):
        model_name += ".onnx"
    if onnx_models_path is None:
        onnx_models_path = get_onnx_models_path()
    model_path = onnx_models_path + model_name

    # Load and check model structure
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    logger = get_std_out_logger()
    logger.info(f"Exported model {model_name} has a valid structure")

    # No need to continue if we cannot compare the models output
    if torch_model is None or model_input is None:
        return

    # This line is a workaround to prevent the following error:
    #   OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Run torch model inference
    torch_out = torch_model(model_input)

    # Prepare ONNX model for inference
    ort_session = onnxruntime.InferenceSession(model_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    logger.info(f"Exported model {model_name} has been tested with ONNXRuntime, and the result looks good!")


def get_onnx_models_path():
    script_path = pathlib.Path(__file__).parent.absolute()
    onnx_models_path = str(script_path) + "/../../onnx_models/"
    return onnx_models_path