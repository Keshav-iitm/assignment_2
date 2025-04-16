import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# thop for computation calculation
try:
    from thop import profile
    thop_available = True
except ImportError:
    if __name__ == '__main__':
        print("Warning: 'thop' library not found. Computation calculation will be skipped.")
        print("Install it using: pip install thop")
    thop_available = False

# Activation Function Mapping
activation_map = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'silu': nn.SiLU,
    'mish': nn.Mish,
    'sigmoid': nn.Sigmoid
}

# CNN Model Definition with Optional BatchNorm and Dropout 
class CNNModel(nn.Module):
    """
    CNN model with optional Batch Normalization and Dropout.
    - 5 Conv blocks: Conv2d -> [BatchNorm2d] -> Activation -> MaxPool2d -> Dropout2d
    - Flatten -> Linear -> Activation -> Dropout -> Linear (Output)
    """
    def __init__(self,
                 num_classes,
                 input_channels,
                 input_height,
                 input_width,
                 filter_counts,
                 kernel_size,
                 activation_name,
                 dense_neurons,
                 dropout_rate,
                 use_batchnorm=False, 
                 pool_kernel_size=2,
                 pool_stride=2):
        """
        Initializes the CNN layers including optional BatchNorm and Dropout.

        Args:
            use_batchnorm (bool): If True, adds BatchNorm2d layers after conv layers.
            (Other args remain the same)
        """
        super(CNNModel, self).__init__()

        # Input Validation
        if not isinstance(num_classes, int) or num_classes <= 0: raise ValueError("num_classes must be > 0")
        if not isinstance(input_height, int) or input_height <= 0: raise ValueError("input_height must be > 0")
        if not isinstance(input_width, int) or input_width <= 0: raise ValueError("input_width must be > 0")
        if not isinstance(filter_counts, (list, tuple)) or len(filter_counts) != 5: raise ValueError("filter_counts must be list/tuple of 5")
        if not all(isinstance(f, int) and f > 0 for f in filter_counts): raise ValueError("filter counts must be positive integers")
        if activation_name not in activation_map: raise ValueError(f"Unsupported activation: {activation_name}")
        if not isinstance(dense_neurons, int) or dense_neurons <= 0: raise ValueError("dense_neurons must be > 0")
        if not (isinstance(dropout_rate, float) and 0.0 <= dropout_rate < 1.0): raise ValueError("dropout_rate must be 0.0 <= p < 1.0")

        # Store Config 
        self.use_batchnorm = use_batchnorm
        self.activation_fn = activation_map[activation_name]()

        # Kernel Size Handling 
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size] * 5
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 5:
            if not all(isinstance(k, int) and k > 0 for k in kernel_size): raise ValueError("kernel sizes must be positive integers")
            kernel_sizes = kernel_size
        else:
            raise ValueError("kernel_size must be an int or a list/tuple of length 5")

        # Layer Definitions 
        self.conv_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList() if self.use_batchnorm else None
        self.pool_layers = nn.ModuleList()
        self.dropout_conv_layers = nn.ModuleList()

        in_channels = input_channels
        current_h, current_w = input_height, input_width

        for i in range(5):
            out_channels = filter_counts[i]
            k_size = kernel_sizes[i]
            use_bias_conv = not self.use_batchnorm
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding='same', bias=use_bias_conv)
            self.conv_layers.append(conv)

            if self.use_batchnorm:
                bn = nn.BatchNorm2d(out_channels)
                self.batchnorm_layers.append(bn)

            pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool_layers.append(pool)

            dropout_conv = nn.Dropout2d(p=dropout_rate)
            self.dropout_conv_layers.append(dropout_conv)

            current_h = (current_h - pool_kernel_size) // pool_stride + 1
            current_w = (current_w - pool_kernel_size) // pool_stride + 1
            if current_h <= 0 or current_w <= 0:
                raise ValueError(f"Image dimensions non-positive after pool layer {i+1}.")

            in_channels = out_channels

        self.flattened_size = in_channels * current_h * current_w
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(self.flattened_size, dense_neurons)
        self.dropout_dense = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(dense_neurons, num_classes)

        if __name__ == '__main__':
            bn_status = "ENABLED" if self.use_batchnorm else "DISABLED"
            print(f"--- CNNModel Initialized (Dropout p={dropout_rate:.2f}, BatchNorm: {bn_status}) ---")
            print(f"  Input: {input_channels}x{input_height}x{input_width}")
        print(f"  Num Classes: {num_classes}")
        print(f"  Conv Filters: {filter_counts}")
        print(f"  Conv Kernels: {kernel_sizes}")
        print(f"  Activation: {activation_name}")
        print(f"  Pooling: Kernel={pool_kernel_size}, Stride={pool_stride}")
        print(f"  Final feature map size before flatten: {in_channels} x {current_h} x {current_w}")
        print(f"  Flattened Size (calculated): {self.flattened_size}")
        print(f"  Dense Layer Neurons: {dense_neurons}")
        print(f"----------------------------")


    def forward(self, x):
        """ Defines the forward pass including optional BatchNorm and Dropout. """
        for i in range(5):
            x = self.conv_layers[i](x)
            if self.batchnorm_layers:
                x = self.batchnorm_layers[i](x)
            x = self.activation_fn(x)
            x = self.pool_layers[i](x)
            x = self.dropout_conv_layers[i](x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.dropout_dense(x)
        x = self.output_layer(x)
        return x

# Calculating trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model definitions and calculations
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CNN Model Def, Param/Computation Calc (Q1 + Dropout + Opt BatchNorm)")
    parser.add_argument('--num_classes', type=int, default=10, help='Num classes (default: 10).')
    parser.add_argument('--input_height', type=int, default=64, help='Input image height (default: 64).')
    parser.add_argument('--input_width', type=int, default=64, help='Input image width (default: 64).')
    parser.add_argument('--input_channels', type=int, default=3, help='Input channels (default: 3).')
    parser.add_argument('--filter_counts', type=int, nargs=5, default=[16, 32, 64, 128, 256], help='Filter counts (default: [16...]).')
    parser.add_argument('--kernel_size', type=int, default=3, help='Conv kernel size (default: 3).')
    parser.add_argument('--activation', type=str, default='relu', choices=list(activation_map.keys()), help='Activation function (default: relu).')
    parser.add_argument('--dense_neurons', type=int, default=128, help='Dense neurons (default: 128).')
    parser.add_argument('--pool_kernel_size', type=int, default=2, help='Pool kernel size (default: 2).')
    parser.add_argument('--pool_stride', type=int, default=2, help='Pool stride (default: 2).')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout probability (default: 0.2)')
    parser.add_argument('--use_batchnorm', action='store_true', help='Use Batch Normalization layers') 

    args = parser.parse_args()
    print("\n*** NOTE: Using the following configuration values ***")
    print(f"*** --use_batchnorm: {args.use_batchnorm} ***")
    print("\n*** NOTE: Using the following configuration values ***")
    print(f"*** --num_classes: {args.num_classes} (Default was 10)")
    print(f"*** --input_height: {args.input_height} (Default was 64)")
    print(f"*** --input_width: {args.input_width} (Default was 64)")
    print("*** Ensure these values are correct for your dataset when training! ***\n")

    def calculate_and_print_stats(model_instance, input_size_tuple, description):
        print(f"\n--- {description} ---")
        try:
            params = count_parameters(model_instance)
            print(f"Parameters: {params:,}")
            if thop_available:
                dummy_input_flops = torch.randn(1, *input_size_tuple)
                model_instance.eval()
                macs, thop_params = profile(model_instance, inputs=(dummy_input_flops,), verbose=False)
                model_instance.train()
                flops = 2 * macs
                print(f"MACs: {macs:,.0f} (~ {macs/1e9:.3f} GigaMACs)")
                print(f"FLOPs (Approx): {flops:,.0f} (~ {flops/1e9:.3f} GigaFLOPs)")
            else:
                print("Computations (MACs/FLOPs): Skipped ('thop' library not installed)")
        except Exception as e:
            print(f"Error during stats calculation: {e}")

    print("\n--- Testing Model Instantiation & Forward Pass (Defaults) ---")
    try:
        model_default = CNNModel(
            num_classes=args.num_classes, input_channels=args.input_channels,
            input_height=args.input_height, input_width=args.input_width,
            filter_counts=args.filter_counts, kernel_size=args.kernel_size,
            activation_name=args.activation, dense_neurons=args.dense_neurons,
            pool_kernel_size=args.pool_kernel_size, pool_stride=args.pool_stride,
            dropout_rate=args.dropout_rate,
            use_batchnorm=args.use_batchnorm 
        )
        # Specific Parameter & Computation Calculations 
        print("\n\n--- Q1 Specific Calculations ---")
        print(f"--- (Using Base: input={args.input_height}x{args.input_width}, classes={args.num_classes}, filters={args.filter_counts}, activation={args.activation}) ---")

        # Scenario 1: k=3, n=128
        try:
            model_k3_n128 = CNNModel(
                num_classes=args.num_classes, input_channels=args.input_channels,
                input_height=args.input_height, input_width=args.input_width,
                filter_counts=args.filter_counts, kernel_size=3, activation_name=args.activation,dropout_rate=args.dropout_rate,
                dense_neurons=128, pool_kernel_size=args.pool_kernel_size, pool_stride=args.pool_stride
            )
            calculate_and_print_stats(
                model_k3_n128,
                (args.input_channels, args.input_height, args.input_width),
                "Q1 Scenario (i): k=3, n=128"
            )
        except Exception as e_k3:
            print(f"\nError initializing or analyzing model for k=3, n=128: {e_k3}")

        # Scenario 2: k=5, n=256
        try:
            model_k5_n256 = CNNModel(
                num_classes=args.num_classes, input_channels=args.input_channels,
                input_height=args.input_height, input_width=args.input_width,
                filter_counts=args.filter_counts, kernel_size=5, activation_name=args.activation,dropout_rate=args.dropout_rate,
                dense_neurons=256, pool_kernel_size=args.pool_kernel_size, pool_stride=args.pool_stride
            )
            calculate_and_print_stats(
                model_k5_n256,
                (args.input_channels, args.input_height, args.input_width),
                "Q1 Scenario (ii): k=5, n=256"
            )
        except Exception as e_k5:
            print(f"\nError initializing or analyzing model for k=5, n=256: {e_k5}")

    except ValueError as ve:
        print(f"\n--- Configuration Error during Model Init ---")
        print(ve)
    except Exception as e:
        print(f"\n--- An unexpected error occurred during main execution ---")
        print(e)
        import traceback
        traceback.print_exc()
        