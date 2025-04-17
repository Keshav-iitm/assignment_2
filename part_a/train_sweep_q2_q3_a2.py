# Creates a NEW WandB sweep with a specific name and parameters,
# Includes meaningful RUN names and robust W&B login check.

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import tqdm
import traceback
import sys
from types import SimpleNamespace
import getpass
import argparse

# Importing custom modules
try:
    # File containing just the CNNModel class definition
    from model_q1_a2 import CNNModel
except ImportError:
    print("Error: Could not import CNNModel from model_q1_a2.py.", file=sys.stderr)
    print("Ensure model_q1_a2.py is in the same directory.", file=sys.stderr)
    sys.exit(1)
try:
    # File containing get_dataloaders with stratified split logic
    from dataset_q2_a2 import get_dataloaders
except ImportError:
    print("Error: Could not import get_dataloaders from dataset_q2_a2.py.", file=sys.stderr)
    print("Ensure dataset_q2_a2.py (with stratified split) is in the same directory.", file=sys.stderr)
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 Argparse Config")

    # Parameters from DEFAULT_CONFIG
    parser.add_argument("--SEED", type=int, default=42, help="Random seed")
    parser.add_argument("--DATASET_ROOT", type=str, default="./inaturalist_12K",
                        help="Path to the iNaturalist dataset root")
    parser.add_argument("--IMAGE_SIZE", type=int, default=224, help="Input image size")

    # WandB settings
    parser.add_argument("--WANDB_ENTITY", type=str,
                        default="ae24s021-indian-institute-of-technology-madras",
                        help="WandB entity name")
    parser.add_argument("--WANDB_PROJECT", type=str,
                        default="DA6401-Assignment2",
                        help="WandB project name")
    parser.add_argument("--NEW_SWEEP_RUN_COUNT", type=int, default=70,
                        help="Number of sweep trials to run")

    # API Key (optional if set in env)
    parser.add_argument("--WANDB_API_KEY", type=str, default=None,
                        help="WandB API Key (or set via env variable WANDB_API_KEY)")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Fallback to env variable for API key
    WANDB_API_KEY = args.WANDB_API_KEY or os.getenv("WANDB_API_KEY")

    if not WANDB_API_KEY:
        raise ValueError("WANDB_API_KEY not found! Set it via --WANDB_API_KEY or as an environment variable.")

    # Login to WandB
    wandb.login(key=WANDB_API_KEY)

    # Reflecting original CAPS naming style
    DEFAULT_CONFIG = {
        "seed": args.SEED,
        "dataset_root": args.DATASET_ROOT,
        "image_size": args.IMAGE_SIZE
    }
    WANDB_ENTITY = args.WANDB_ENTITY
    WANDB_PROJECT = args.WANDB_PROJECT
    NEW_SWEEP_RUN_COUNT = args.NEW_SWEEP_RUN_COUNT

    print(" Config loaded:")
    print(f"  DEFAULT_CONFIG: {DEFAULT_CONFIG}")
    print(f"  WANDB_ENTITY: {WANDB_ENTITY}")
    print(f"  WANDB_PROJECT: {WANDB_PROJECT}")
    print(f"  NEW_SWEEP_RUN_COUNT: {NEW_SWEEP_RUN_COUNT}")
    print(f"  WANDB_API_KEY: {'Provided' if WANDB_API_KEY else 'Missing'}")

# Define the NEW sweep configuration
sweep_configuration = {
    'method': 'bayes', # Using Bayesian Optimization
    'name': 'LocalBayesSweep_v2', 
    'metric': { 'name': 'val_accuracy', 'goal': 'maximize' }, # Optimize for validation accuracy
    'parameters': {
        # Define search space for hyperparameters
        'epochs': { 'values': [10, 15] },
        'batch_size': { 'values': [8] },
        'learning_rate': { 'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.01 },
        'optimizer': { 'values': ['adam', 'sgd'] },
        'dropout_rate': { 'distribution': 'uniform', 'min': 0.1, 'max': 0.5 },
        'activation': { 'values': ['relu','gelu','silu','mish'] },
        'dense_neurons': { 'values': [128, 256] },
        'use_batchnorm': { 'values': [True, False] },
        'use_augmentation': { 'values': [True, False] },
        'filter_counts' : {'values':[(16,16,16,32,32),(32,32,16,16,16),(16,16,32,32,64),(64,32,32,16)]}
        # kernel_size is fixed at 3 (default in CNNModel)
    }
}

# Helper Functions & Training Logic

# Accuracy Calculation
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1); total = labels.size(0); correct = (predicted == labels).sum().item()
    return (correct / total) if total > 0 else 0.0

# Training/Validation Functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train(); running_loss = 0.0; running_accuracy = 0.0; batch_count = 0
    pbar = tqdm.tqdm(dataloader, desc="Train", leave=False, unit="batch", file=sys.stdout)
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
        loss. backward(); optimizer.step(); running_loss += loss.item(); batch_accuracy = calculate_accuracy(outputs, labels)
        running_accuracy += batch_accuracy; batch_count += 1
        pbar.set_postfix({'L':f'{running_loss / batch_count:.3f}','A':f'{running_accuracy / batch_count:.3f}'})
    return (running_loss / batch_count, running_accuracy / batch_count) if batch_count > 0 else (0.0, 0.0)

def validate(model, dataloader, criterion, device):
    model.eval(); running_loss = 0.0; running_accuracy = 0.0; batch_count = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc="Val", leave=False, unit="batch", file=sys.stdout)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device); outputs = model(inputs); loss = criterion(outputs, labels)
            running_loss += loss.item(); batch_accuracy = calculate_accuracy(outputs, labels)
            running_accuracy += batch_accuracy; batch_count += 1
            pbar.set_postfix({'vL':f'{running_loss / batch_count:.3f}','vA':f'{running_accuracy / batch_count:.3f}'})
    return (running_loss / batch_count, running_accuracy / batch_count) if batch_count > 0 else (0.0, 0.0)

# Main Training Logic function (executed by wandb.agent)
def train_logic_for_agent():
    """ Function defining the training steps for a single sweep trial """
    best_val_accuracy = 0.0
    run = None # Initialize run variable
    finish_called_in_try = False
    try:
        # wandb.init() here connects to the agent's managed run context
        run = wandb.init()
        config = wandb.config
        if run is None: raise RuntimeError("wandb.init() failed within agent context.")
        config_ns = SimpleNamespace(**dict(config)) # Use dot notation

        # Start of Trial 
        print(f"\n--- Starting Trial {wandb.run.id} for Sweep {run.sweep_id} ---", file=sys.stdout)
        print(f"Hyperparameters: {vars(config_ns)}", file=sys.stdout)

        # Setup: Seed, Device
        torch.manual_seed(DEFAULT_CONFIG['seed']); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(DEFAULT_CONFIG['seed'])
        print(f"Device: {device}", file=sys.stdout)

        # Generate and set Meaningful RUN Name
        try:
             run_name = (f"opt={config_ns.optimizer}_lr={config_ns.learning_rate:.4f}_bs={config_ns.batch_size}_"
                         f"do={config_ns.dropout_rate:.2f}_act={config_ns.activation}_dn={config_ns.dense_neurons}_"
                         f"bn={str(config_ns.use_batchnorm)[0]}_aug={str(config_ns.use_augmentation)[0]}_"
                         f"e={config_ns.epochs}_fil={config_ns.filter_counts}")
             wandb.run.name = run_name.replace("True", "T").replace("False", "F")
             wandb.run.save() # Explicitly save the name change
             print(f"Set Run Name: {wandb.run.name}", file=sys.stdout)
        except Exception as name_e: print(f"Warning: Error setting run name: {name_e}", file=sys.stderr)

        # Load Data
        print(f"Loading data from: {DEFAULT_CONFIG['dataset_root']}", file=sys.stdout)
        train_loader, val_loader, _, num_classes, _ = get_dataloaders(
            dataset_root=DEFAULT_CONFIG['dataset_root'], batch_size=config_ns.batch_size,
            use_augmentation=config_ns.use_augmentation,
            random_seed=DEFAULT_CONFIG['seed'])
        if train_loader is None or val_loader is None or num_classes <= 0: raise ValueError("Data loading failed.")
        print(f"Data loaded: {num_classes} classes.", file=sys.stdout)

        # Create Model
        print("Creating model...", file=sys.stdout); model = CNNModel( num_classes=num_classes, input_channels= 3, 
            input_height = DEFAULT_CONFIG['image_size'],
                 input_width = DEFAULT_CONFIG['image_size'],
                 filter_counts=config_ns.filter_counts,
                 kernel_size=3,
                activation_name=config_ns.activation,
            dense_neurons=config_ns.dense_neurons, dropout_rate=config_ns.dropout_rate, use_batchnorm=config_ns.use_batchnorm)
        model.to(device); print("Model created.", file=sys.stdout)

        # Setup Optimizer, Loss, Scheduler
        print(f"Setting up: {config_ns.optimizer}, LR: {config_ns.learning_rate}", file=sys.stdout)
        if config_ns.optimizer.lower() == "adam": optimizer = optim.Adam(model.parameters(), lr=config_ns.learning_rate)
        elif config_ns.optimizer.lower() == "sgd": optimizer = optim.SGD(model.parameters(), lr=config_ns.learning_rate, momentum=0.9)
        else: raise ValueError(f"Unsupported optimizer '{config_ns.optimizer}'")
        criterion = nn.CrossEntropyLoss(); scheduler = CosineAnnealingLR(optimizer, T_max=config_ns.epochs, eta_min=1e-6)
        print("Optimizer, Loss, Scheduler OK.", file=sys.stdout)

        # WandB Watch Model
        wandb.watch(model, criterion, log="all", log_freq=len(train_loader) // 2)

        # Training Loop
        print(f"\nTraining for {config_ns.epochs} epochs...", file=sys.stdout)
        for epoch in range(config_ns.epochs):
            epoch_num = epoch + 1
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            current_lr = scheduler.get_last_lr()[0]
            # Print progress
            print(f"E{epoch_num}/{config_ns.epochs} LR:{current_lr:.6f}|Tr L:{train_loss:.4f},A:{train_acc:.4f}|Val L:{val_loss:.4f},A:{val_acc:.4f}", file=sys.stdout)
            sys.stdout.flush()
            # Log metrics to WandB
            wandb.log({"epoch": epoch_num, "learning_rate": current_lr, "train_loss": train_loss,
                       "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc })
            scheduler.step()
            # Save checkpoint to WandB Files
            if val_acc > best_val_accuracy:
                 best_val_accuracy = val_acc; print(f"*** New best: {best_val_accuracy:.4f}. Saving... ***", file=sys.stdout)
                 save_path = f'best_model_{wandb.run.id}_e{epoch_num}.pth' # Local temp name
                 try:
                     torch.save(model.state_dict(), save_path)
                     wandb.save(save_path, base_path=".") # Upload to WandB Files
                     print(f"Checkpoint saved & uploaded.", file=sys.stdout)
                 except Exception as save_e: print(f"Error saving/uploading checkpoint: {save_e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
                 sys.stdout.flush()

        print("\n--- Trial Finished Successfully ---", file=sys.stdout)
        if run: run.finish(); finish_called_in_try = True # Signal successful finish to WandB

    # --- Error Handling for the Trial ---
    except Exception as e:
        print(f"\n!!! Error in trial {wandb.run.id if run else 'UNKNOWN'}: {e} !!!", file=sys.stderr); traceback.print_exc(file=sys.stderr)
        if run and not finish_called_in_try: # Mark run as failed in WandB
             try: run.finish(exit_code=1); finish_called_in_try = True
             except Exception as finish_err: print(f"Error calling finish(exit_code=1): {finish_err}", file=sys.stderr)
    finally: # Final cleanup checks for the trial
        if run and not finish_called_in_try: print("Warning: finish() not called.", file=sys.stderr)
        sys.stdout.flush(); sys.stderr.flush()

# --- Main Script Execution Block ---
if __name__ == '__main__':
    print("--- Creating and Running NEW W&B Sweep (Local Agent Script) ---", file=sys.stdout)

    # Set WANDB_DISABLE_SERVICE env var to potentially prevent agent cleanup errors
    os.environ['WANDB_DISABLE_SERVICE'] = 'true'
    print(f"Set WANDB_DISABLE_SERVICE=true", file=sys.stdout)

    # Robust W&B Login Check/Attempt
    try:
        # Try login using Env Var or netrc first (non-interactive)
        login_success = wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=False, anonymous="never")
        if not login_success:
            print("W&B key not found non-interactively. Attempting interactive login...", file=sys.stdout)
            # If the above fails (returns False), try interactive login which will prompt for key
            if not wandb.login(relogin=True): # Force prompt if needed
                 print("Interactive W&B login failed. Please check your API key or run 'wandb login' manually.", file=sys.stderr)
                 sys.exit(1)
        print("W&B login successful.", file=sys.stdout)
    except Exception as login_e:
        print(f"Error during W&B login process: {login_e}", file=sys.stderr)
        print("Ensure you can log in (e.g., run 'wandb login' in terminal).", file=sys.stderr)
        sys.exit(1)
    sys.stdout.flush() # Ensure login status is printed

    # 1. Create the NEW Sweep on WandB servers
    try:
        print(f"Creating new sweep '{sweep_configuration.get('name', 'Unnamed')}' in project '{WANDB_PROJECT}'...", file=sys.stdout)
        new_sweep_id = wandb.sweep(
            sweep=sweep_configuration, # The configuration dict defined above
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT
        )
        print(f"Successfully created new sweep with ID: {new_sweep_id}", file=sys.stdout)
        sweep_id_full = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{new_sweep_id}" # Construct full path for agent
    except Exception as sweep_e:
        print(f"!!! Failed to create sweep: {sweep_e} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    sys.stdout.flush()

    # 2. Start the agent for the NEWLY created sweep
    print(f"\nStarting agent for NEW sweep: {sweep_id_full}", file=sys.stdout)
    print(f"Agent will attempt up to {NEW_SWEEP_RUN_COUNT} trials.", file=sys.stdout)
    try:
        # This call blocks until the agent finishes its count or the sweep completes/errors
        wandb.agent(
            sweep_id=sweep_id_full,         # Full sweep path from wandb.sweep()
            function=train_logic_for_agent, # The function defining work for one trial
            count=NEW_SWEEP_RUN_COUNT       # Max trials for this script execution
        )
        print("\n--- wandb.agent finished its work (completed count, sweep finished, or stopped) ---", file=sys.stdout)
    except Exception as agent_e:
        # Catch errors from the agent process itself (e.g., connection issues)
        print(f"\n!!! Error running wandb.agent: {agent_e} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        # Ensure all output is flushed before exiting
        sys.stdout.flush(); sys.stderr.flush()

    print("\n--- Sweep Creation and Agent Script Completed ---", file=sys.stdout)

