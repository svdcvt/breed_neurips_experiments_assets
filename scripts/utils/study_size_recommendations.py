import argparse
from utility import get_available_memory, calculate_data_size

def main():
    parser = argparse.ArgumentParser(description='Calculate data size and maximum training samples')
    parser.add_argument('--space-points', '-N', type=int, required=True, help='Number of space discretization points')
    parser.add_argument('--time-points', '-T', type=int, required=True, help='Number of time discretization points')
    parser.add_argument('--val-samples', '-D', type=int, required=True, help='Number of validation samples')
    parser.add_argument('--val-time-points', '-VT', type=int, help='Number of time points for validation (if different)')
    parser.add_argument('--precision', '-P', type=int, choices=[32, 64], default=32, help='Precision in bits')
    parser.add_argument('--memory', '-M', type=float, help='Available memory in GB (if not specified, system memory is used)')
    parser.add_argument('--zmq-hwm', '-H', type=float, default=0.05, help='ZeroMQ high water mark (default: 0.05) percentage of buffer size')

    args = parser.parse_args()
    
    # Use provided validation time points or default to training time points
    val_time_points = args.val_time_points if args.val_time_points else args.time_points
    
    # Calculate validation set size
    val_size = calculate_data_size(args.space_points, val_time_points, args.val_samples, args.precision)
    
    # Get available memory
    if args.memory:
        available_memory = args.memory * 1024**3  # Convert GB to bytes
    else:
        available_memory = get_available_memory()
    
    # Calculate maximum training samples possible
    memory_for_training = available_memory - val_size
    sample_size = calculate_data_size(args.space_points, args.time_points, 1, args.precision)
    buffer_item_size = calculate_data_size(args.space_points, 1, 2, args.precision)
    max_training_samples = int(round(memory_for_training / sample_size))
    max_buffer_size = int(round(memory_for_training / buffer_item_size))


    # Print results
    print(f"\nData Size Estimation:")
    print(f"Validation set size: {val_size / 1024**3:.2f} GB")
    print(f"Available memory: {available_memory / 1024**3:.2f} GB")
    print(f"One sample size: {sample_size / 1024**2:.2f} MB")
    print(f"Maximum training samples possible: {max_training_samples}")
    print(f"Required memory for max training set: {(max_training_samples * sample_size) / 1024**3:.2f} GB")
    print(f"Buffer item size (t, t+1): {buffer_item_size / 1024**2:.2f} MB")
    print(f"Maximum buffer size possible: {max_buffer_size}")
    print(f"Required memory for max buffer size: {(max_buffer_size * buffer_item_size) / 1024**3:.2f} GB")
    print(f"Percentage of zmq_hwm in buffer: {args.zmq_hwm}")
    print(f"Recommendation: buffer size = {int(round(max_buffer_size*(1-args.zmq_hwm)))}, zmq_hwm = {int(round(max_buffer_size*args.zmq_hwm))}")
    

if __name__ == "__main__":
    main()