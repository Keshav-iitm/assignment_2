import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# --- NEW: Import thop ---
try:
    from thop import profile
    from thop import clever_format # Optional: for cleaner output formatting
    thop_available = True
except ImportError:
    print("Warning: 'thop' library not found. Computation calculation (MACs/FLOPs) will be skipped.")
    print("Install it using: pip install thop")
    thop_available = False

# --- Activation Function Mapping ---
activation_map = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}

# --- CNN Model Definition ---
class CNNModel(nn.Module):
    """
    Convolutional Neural Network model as specified in Question 1.
    (Docstring remains the same)
    """
    def __init__(self,
                 num_classes,
                 input_channels,
                 input_height,
                 input_width,
                 filter_counts,
                 kernel_size, # Can be single int or list/tuple of 5 ints
                 activation_name, # Name of the activation function (e.g., 'relu')
                 dense_neurons,
                 pool_kernel_size=2,
                 pool_stride=2):
        super(CNNModel, self).__init__()
        # --- Input Validation ---
        if not isinstance(num_classes, int) or num_classes <= 0:
             raise ValueError("num_classes must be a positive integer.")
        if not isinstance(input_height, int) or input_height <= 0:
             raise ValueError("input_height must be a positive integer.")
        if not isinstance(input_width, int) or input_width <= 0:
             raise ValueError("input_width must be a positive integer.")
        if not isinstance(filter_counts, (list, tuple)) or len(filter_counts) != 5:
            raise ValueError("filter_counts must be a list/tuple of length 5")
        if not all(isinstance(f, int) and f > 0 for f in filter_counts):
            raise ValueError("All filter counts must be positive integers.")
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation_name}. "
                             f"Available: {list(activation_map.keys())}")
        if not isinstance(dense_neurons, int) or dense_neurons <= 0:
            raise ValueError("dense_neurons must be a positive integer.")

        # --- Activation Function ---
        self.activation_fn = activation_map[activation_name]()

        # --- Kernel Size Handling ---
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size] * 5
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 5:
             if not all(isinstance(k, int) and k > 0 for k in kernel_size):
                 raise ValueError("All kernel sizes must be positive integers.")
             kernel_sizes = kernel_size
        else:
            raise ValueError("kernel_size must be an int or a list/tuple of length 5")

        # --- Layer Definitions ---
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        in_channels = input_channels
        current_h, current_w = input_height, input_width
        for i in range(5):
            out_channels = filter_counts[i]
            k_size = kernel_sizes[i]
            if k_size % 2 == 0:
                 print(f"Warning: Even kernel size ({k_size}) used in layer {i+1}.")
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding='same', bias=True)
            self.conv_layers.append(conv)
            pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool_layers.append(pool)
            current_h = (current_h - pool_kernel_size) // pool_stride + 1
            current_w = (current_w - pool_kernel_size) // pool_stride + 1
            if current_h <= 0 or current_w <= 0:
                raise ValueError(f"Image dimensions non-positive ({current_h}x{current_w}) after pool layer {i+1}.")
            in_channels = out_channels

        self.flattened_size = in_channels * current_h * current_w
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(self.flattened_size, dense_neurons)
        self.output_layer = nn.Linear(dense_neurons, num_classes)

        print(f"--- CNNModel Initialized ---")
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
        """ Defines the forward pass of the model. """
        for i in range(5):
            x = self.conv_layers[i](x)
            x = self.activation_fn(x)
            x = self.pool_layers[i](x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.output_layer(x)
        return x

# --- Function to Calculate Trainable Parameters ---
def count_parameters(model):
    """ Counts the total number of trainable parameters in the model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Main block for testing model definition and Q1 calculations ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CNN Model Definition, Parameter and Computation Calculation (Q1)")

    # Arguments remain the same...
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes (default: 10).')
    parser.add_argument('--input_height', type=int, default=64, help='Height of input images (default: 64).')
    parser.add_argument('--input_width', type=int, default=64, help='Width of input images (default: 64).')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input image channels (default: 3).')
    parser.add_argument('--filter_counts', type=int, nargs=5, default=[16, 32, 64, 128, 256], help='List of 5 filter counts (default: [16...]).')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for conv layers (default: 3).')
    parser.add_argument('--activation', type=str, default='relu', choices=list(activation_map.keys()), help='Activation function name (default: relu).')
    parser.add_argument('--dense_neurons', type=int, default=128, help='Neurons in hidden dense layer (default: 128).')
    parser.add_argument('--pool_kernel_size', type=int, default=2, help='Max pooling kernel size (default: 2).')
    parser.add_argument('--pool_stride', type=int, default=2, help='Max pooling stride (default: 2).')

    args = parser.parse_args()

    print("\n*** NOTE: Using the following configuration values ***")
    print(f"*** --num_classes: {args.num_classes} (Default was 10)")
    print(f"*** --input_height: {args.input_height} (Default was 64)")
    print(f"*** --input_width: {args.input_width} (Default was 64)")
    print("*** Ensure these values are correct for your dataset when training! ***\n")

    # --- Helper function for calculation ---
    def calculate_and_print_stats(model, input_size_tuple, description):
        print(f"\n--- {description} ---")
        try:
            # Parameter Calculation
            params = count_parameters(model)
            print(f"Parameters: {params:,}")

            # Computation Calculation (if thop available)
            if thop_available:
                # Use batch size 1 for FLOPs calculation
                dummy_input_flops = torch.randn(1, *input_size_tuple)
                macs, thop_params = profile(model, inputs=(dummy_input_flops,), verbose=False)

                # FLOPs are typically ~2x MACs
                flops = 2 * macs

                # Use clever_format for readability if desired
                # macs_str, params_str = clever_format([macs, thop_params], "%.3f")
                # flops_str, _ = clever_format([flops, thop_params], "%.3f")
                # print(f"MACs: {macs_str}")
                # print(f"FLOPs (Approx): {flops_str}")

                # Or print raw numbers (often preferred for reports)
                print(f"MACs: {macs:,.0f} (~ {macs/1e9:.3f} GigaMACs)")
                print(f"FLOPs (Approx): {flops:,.0f} (~ {flops/1e9:.3f} GigaFLOPs)")

                # Sanity check: Compare parameter counts
                if abs(params - thop_params) > 10: # Allow small diff due to potential calculation nuances
                     print(f"Warning: Parameter count mismatch! count_parameters: {params:,}, thop: {thop_params:,}")
            else:
                 print("Computations (MACs/FLOPs): Skipped ('thop' library not installed)")

        except ValueError as ve:
            print(f"Configuration Error during stats calculation: {ve}")
        except Exception as e:
            print(f"Error during stats calculation: {e}")
            import traceback
            traceback.print_exc()

    # --- Main Test and Calculations ---
    print("\n--- Testing Model Instantiation & Forward Pass (Defaults) ---")
    try:
        model_default = CNNModel(
            num_classes=args.num_classes, input_channels=args.input_channels,
            input_height=args.input_height, input_width=args.input_width,
            filter_counts=args.filter_counts, kernel_size=args.kernel_size,
            activation_name=args.activation, dense_neurons=args.dense_neurons,
            pool_kernel_size=args.pool_kernel_size, pool_stride=args.pool_stride
        )

        # Test forward pass
        dummy_input = torch.randn(2, args.input_channels, args.input_height, args.input_width)
        output = model_default(dummy_input)
        print(f"\nForward pass output shape: {output.shape}")
        if output.shape[0] == 2 and output.shape[1] == args.num_classes:
            print("Forward pass shape test: PASSED")
        else:
            print("Forward pass shape test: FAILED")

        # Calculate stats for the default model configuration
        calculate_and_print_stats(
            model_default,
            (args.input_channels, args.input_height, args.input_width),
            "Stats for Default Configuration (k={}, n={})".format(args.kernel_size, args.dense_neurons)
        )

        # --- Specific Parameter & Computation Calculations for Question 1 ---
        print("\n\n--- Q1 Specific Calculations ---")
        print(f"--- (Using Base: input={args.input_height}x{args.input_width}, classes={args.num_classes}, filters={args.filter_counts}, activation={args.activation}) ---")

        # Scenario 1: k=3, n=128
        try:
            model_k3_n128 = CNNModel(
                num_classes=args.num_classes, input_channels=args.input_channels,
                input_height=args.input_height, input_width=args.input_width,
                filter_counts=args.filter_counts, kernel_size=3, activation_name=args.activation,
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
                filter_counts=args.filter_counts, kernel_size=5, activation_name=args.activation,
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

