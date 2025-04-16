import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

<<<<<<< HEAD
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
=======
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
>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
activation_map = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
<<<<<<< HEAD
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
=======
    'sigmoid': nn.Sigmoid
}

# --- CNN Model Definition ---
class CNNModel(nn.Module):
    """
    Convolutional Neural Network model as specified in Question 1.
    (Docstring remains the same)
>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
    """
    def __init__(self,
                 num_classes,
                 input_channels,
                 input_height,
                 input_width,
                 filter_counts,
<<<<<<< HEAD
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

=======
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
>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
            in_channels = out_channels

        self.flattened_size = in_channels * current_h * current_w
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(self.flattened_size, dense_neurons)
<<<<<<< HEAD
        self.dropout_dense = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(dense_neurons, num_classes)

        if __name__ == '__main__':
            bn_status = "ENABLED" if self.use_batchnorm else "DISABLED"
            print(f"--- CNNModel Initialized (Dropout p={dropout_rate:.2f}, BatchNorm: {bn_status}) ---")
            print(f"  Input: {input_channels}x{input_height}x{input_width}")
=======
        self.output_layer = nn.Linear(dense_neurons, num_classes)

        print(f"--- CNNModel Initialized ---")
        print(f"  Input: {input_channels}x{input_height}x{input_width}")
>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
        print(f"  Num Classes: {num_classes}")
        print(f"  Conv Filters: {filter_counts}")
        print(f"  Conv Kernels: {kernel_sizes}")
        print(f"  Activation: {activation_name}")
        print(f"  Pooling: Kernel={pool_kernel_size}, Stride={pool_stride}")
        print(f"  Final feature map size before flatten: {in_channels} x {current_h} x {current_w}")
        print(f"  Flattened Size (calculated): {self.flattened_size}")
        print(f"  Dense Layer Neurons: {dense_neurons}")
        print(f"----------------------------")

<<<<<<< HEAD

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
=======
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

>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
    print("\n*** NOTE: Using the following configuration values ***")
    print(f"*** --num_classes: {args.num_classes} (Default was 10)")
    print(f"*** --input_height: {args.input_height} (Default was 64)")
    print(f"*** --input_width: {args.input_width} (Default was 64)")
    print("*** Ensure these values are correct for your dataset when training! ***\n")

<<<<<<< HEAD
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

=======
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
>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
    print("\n--- Testing Model Instantiation & Forward Pass (Defaults) ---")
    try:
        model_default = CNNModel(
            num_classes=args.num_classes, input_channels=args.input_channels,
            input_height=args.input_height, input_width=args.input_width,
            filter_counts=args.filter_counts, kernel_size=args.kernel_size,
            activation_name=args.activation, dense_neurons=args.dense_neurons,
<<<<<<< HEAD
            pool_kernel_size=args.pool_kernel_size, pool_stride=args.pool_stride,
            dropout_rate=args.dropout_rate,
            use_batchnorm=args.use_batchnorm 
        )
        # Specific Parameter & Computation Calculations 
=======
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
>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
        print("\n\n--- Q1 Specific Calculations ---")
        print(f"--- (Using Base: input={args.input_height}x{args.input_width}, classes={args.num_classes}, filters={args.filter_counts}, activation={args.activation}) ---")

        # Scenario 1: k=3, n=128
        try:
            model_k3_n128 = CNNModel(
                num_classes=args.num_classes, input_channels=args.input_channels,
                input_height=args.input_height, input_width=args.input_width,
<<<<<<< HEAD
                filter_counts=args.filter_counts, kernel_size=3, activation_name=args.activation,dropout_rate=args.dropout_rate,
=======
                filter_counts=args.filter_counts, kernel_size=3, activation_name=args.activation,
>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
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
<<<<<<< HEAD
                filter_counts=args.filter_counts, kernel_size=5, activation_name=args.activation,dropout_rate=args.dropout_rate,
=======
                filter_counts=args.filter_counts, kernel_size=5, activation_name=args.activation,
>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
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
<<<<<<< HEAD
        
=======

>>>>>>> 69169f1f4e81fc21d38f04e217a4d06d97da3a7a
