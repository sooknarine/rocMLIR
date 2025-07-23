"""
This script compiles a given config in order to collect the following data
points for each config:
- Blocksize
- Gridsize
- vgpr
- sgpr
- LDS allocated
- Occupancy

The given config is expected to be a csv with the following format:
|# arch| numCUs | testVector | perfConfig (exhaustive) |

Usage:
    python3 compileAndCollectTuningData.py --op <operation> <config.csv>
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile

from datetime import datetime
from testing_metrics import calculateOccupancy


# TODO use AmdArchDb.py (when it's implemented). 4 works for all current
# architectures, but this may not hold in the future.
numEUPerCU = 4

class TuningData:
    """Class to represent tuning data results."""
    
    def __init__(self):
        self.blocksize = None
        self.gridsize = None
        self.vgpr_count = None
        self.vgpr_spills = None
        self.sgpr_count = None
        self.sgpr_spills = None
        self.LDS_allocated = None
        self.occupancy = None
    
    def to_dict(self):
        """Convert to dictionary format for CSV writing."""
        return {
            'blocksize': self.blocksize,
            'gridsize': self.gridsize,
            'vgpr_count': self.vgpr_count,
            'vgpr_spills': self.vgpr_spills,
            'sgpr_count': self.sgpr_count,
            'sgpr_spills': self.sgpr_spills,
            'LDS_allocated': self.LDS_allocated,
            'occupancy': self.occupancy
        }

def create_tuning_data():
    return TuningData()

def check_rocmlir_binaries():
    """
    Check if rocmlir-gen and rocmlir-driver binaries can be found.
    Returns the paths to the binaries if found, otherwise exits.
    """

    # Define expected binary paths. This scripts expects that all of the
    # scripts have already been built (using ninja ci-performance-scripts) and
    # are located in the build/bin directory
    build_bin_dir = os.path.dirname(os.path.abspath(__file__))
    rocmlir_gen_path = os.path.join(build_bin_dir, 'rocmlir-gen')
    rocmlir_driver_path = os.path.join(build_bin_dir, 'rocmlir-driver')

    # Check if build/bin directory exists
    if not os.path.exists(build_bin_dir):
        print(f"Error: Build directory not found at {build_bin_dir}")
        print("Please make sure the rocMLIR build directory exists.")
        sys.exit(1)

    # Check for rocmlir-gen
    if not os.path.exists(rocmlir_gen_path):
        print(f"Error: rocmlir-gen not found at {rocmlir_gen_path}")
        sys.exit(1)

    # Check for rocmlir-driver
    if not os.path.exists(rocmlir_driver_path):
        print(f"Error: rocmlir-driver not found at {rocmlir_driver_path}")
        sys.exit(1)

    print(f"✓ Found rocmlir-gen: {rocmlir_gen_path}")
    print(f"✓ Found rocmlir-driver: {rocmlir_driver_path}")

    return [rocmlir_gen_path, rocmlir_driver_path]

def parse_config_csv(config_file):
    """Parse the input CSV file containing configuration data."""
    configs = []

    try:
        with open(config_file, 'r') as csvfile:
            # Use csv.DictReader with tab delimiter
            reader = csv.DictReader(csvfile, delimiter='\t')
            
            # Process each row
            for row in reader:
                # Strip whitespace from all values
                clean_row = {key.strip(): value.strip() for key,
                                          value in row.items()}
                
                # Only add non-empty rows (check if any value is non-empty)
                if any(clean_row.values()):
                    configs.append(clean_row)
        
        return configs
        
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

def parse_test_args(test_vector):
    """
    Parse test vector arguments to handle boolean flags.
    For patterns like '-flag', 'false' - remove both
    For patterns like '-flag', 'true' - keep only the flag
    For other patterns - keep as is
    """
    args = test_vector.split()
    parsed_args = []
    i = 0

    while i < len(args):
        current_arg = args[i]

        # Check if this is a flag (starts with '-') and has a next argument
        if current_arg.startswith('-') and i + 1 < len(args):
            next_arg = args[i + 1]

            # Check if the next argument is a boolean value
            if next_arg.lower() in ['true', 'false']:
                # Only include the flag if the value is 'true'
                if next_arg.lower() == 'true':
                    parsed_args.append(current_arg)
                # Skip both the flag and the boolean value
                i += 2
            else:
                # Not a boolean flag, keep both arguments
                parsed_args.append(current_arg)
                parsed_args.append(next_arg)
                i += 2
        else:
            # Single argument or last argument, keep as is
            parsed_args.append(current_arg)
            i += 1

    return parsed_args

def convertConvTestArgs(test_args, operation):
    """
    Convert test arguments for convolution operations to the format expected
    by rocmlir-gen.
    
    Args:
        test_args: List of test arguments parsed from the test vector.
    
    Returns:
        List of converted test arguments.
    """
    # Build converted arguments list
    converted_args = []
    
    # Process arguments in pairs
    i = 0
    while i < len(test_args):
        if i == 0:
            # The first argument contains the operation type
            dataType = None
            if test_args[0] == 'conv':
                dataType = 'f32'
            elif test_args[0] == 'convfp16':
                dataType = 'f16'
            elif test_args[0] == 'convbfp16':
                dataType = 'bf16'
            elif test_args[0] == 'convint8':
                dataType = 'i8'
            elif test_args[0] == 'convfp8_fp8':
                dataType = 'fp8_fp8'
            elif test_args[0] == 'convfp8':
                dataType = 'fp8'
            elif test_args[0] == 'convfp8_bf8':
                dataType = 'fp8_bf8'
            elif test_args[0] == 'convbf8_fp8':
                dataType = 'bf8_fp8'
            elif test_args[0] == 'convbf8_bf8':
                dataType = 'bf8_bf8'
            converted_args.extend(["-t", dataType])
            i += 1
            continue

        if i + 1 < len(test_args):
            opt = test_args[i]
            val = test_args[i + 1]
            
            # Map short form arguments to rocmlir-gen long form
            if opt == "-n":
                converted_args.extend(["--batchsize", val])
            elif opt == "-c":
                converted_args.extend(["--in_channels", val])
            elif opt == "-H":
                converted_args.extend(["--in_h", val])
            elif opt == "-W":
                converted_args.extend(["--in_w", val])
            elif opt == "-k":
                converted_args.extend(["--out_channels", val])
            elif opt == "-y":
                converted_args.extend(["--fil_h", val])
            elif opt == "-x":
                converted_args.extend(["--fil_w", val])
            elif opt == "-p":
                converted_args.extend(["--padding_h", val])
            elif opt == "-q":
                converted_args.extend(["--padding_w", val])
            elif opt == "-u":
                converted_args.extend(["--conv_stride_h", val])
            elif opt == "-v":
                converted_args.extend(["--conv_stride_w", val])
            elif opt == "-l":
                converted_args.extend(["--dilation_h", val])
            elif opt == "-j":
                converted_args.extend(["--dilation_w", val])
            elif opt == "-g":
                converted_args.extend(["-g", val])
            elif opt == "-f":
                converted_args.extend(["--fil_layout", val.lower()])
            elif opt == "-I":
                converted_args.extend(["--in_layout", val.lower()])
            elif opt == "-O":
                converted_args.extend(["--out_layout", val.lower()])
            elif opt == "-F":
                # Convert direction flag to operation
                direction_val = int(val)
                if direction_val == 1:
                    operation = "conv"
                elif direction_val == 2:
                    operation = "conv_bwd_data"
                elif direction_val == 4:
                    operation = "conv_bwd_weight"
            else:
                # Unknown argument, do not add
                pass
            i += 2
    
    return [converted_args, operation]

def compile_config(config, operation, binaries, timestamp):
    arch = config["# arch"].split(':')[0]
    num_cu = config["numCUs"]

    # Parse and add the test vector arguments
    test_vector = config["testVector"]
    test_args = parse_test_args(test_vector)

    # If operation is a convolution, we need to convert the test_args to a
    # format that rocmlir-gen can understand
    if operation.lower() == 'conv':
        [test_args, operation] = convertConvTestArgs(test_args, operation)

    # Build the rocmlir-gen command
    rocmlir_gen_cmd = [
        binaries[0],
        "--operation", operation,
        "--arch", arch,
        "--num_cu", num_cu,
    ]
    
    rocmlir_gen_cmd.extend(test_args)

    # Add perf_config
    rocmlir_gen_cmd.extend(["--perf_config",
                            config["perfConfig (exhaustive)"]])
    
    # Add output file
    rocmlir_gen_cmd.extend(["-o",
                            f"rocmlir-gen-output-{arch}-{timestamp}.mlir"])

    # Build the rocmlir-driver command
    rocmlir_driver_cmd = [
        binaries[1],
        "-c",
        f"--arch={arch}",
        "--debug-only=convert-rock-to-gpu,serialize-to-isa",
        f"rocmlir-gen-output-{arch}-{timestamp}.mlir",
        "-o", f"rocmlir-driver-output-{arch}-{timestamp}.mlir"
    ]

    commands = [rocmlir_gen_cmd, rocmlir_driver_cmd]

    # Execute commands sequentially
    try:
        for i, cmd in enumerate(commands):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # We want to write the output from the rocmlir-driver command
            # to a text file so that we can save the debug output
            if i == 1:
                # Write debug output to file
                debug_file = f"rocmlir-driver-debug-{arch}-{timestamp}.txt"
                with open(debug_file, 'w') as f:
                    f.write(result.stderr)


    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None
    
    return [
        f"rocmlir-gen-output-{arch}-{timestamp}.mlir",
        f"rocmlir-driver-output-{arch}-{timestamp}.mlir",
        f"rocmlir-driver-debug-{arch}-{timestamp}.txt",
    ]

def parse_driver_debug_results(tuning_data, dbg_message_file):
    # Parse the rocmlir-driver debug output file for gridsize, blocksize,
    # lds usage, SGPR, and VGPR information
    if os.path.exists(dbg_message_file):
        try:
            with open(dbg_message_file, 'r') as f:
                content = f.read()
                
                # Look for blocksize
                blocksize_match = re.search(r'blockSize:\s*(\d+)', content)
                if blocksize_match:
                    tuning_data.blocksize = int(blocksize_match.group(1))

                # Look for gridsize
                gridsize_match = re.search(r'gridSize:\s*(\d+)', content)
                if gridsize_match:
                    tuning_data.gridsize = int(gridsize_match.group(1))

                # Look for LDS_allocated
                lds_match = re.search(r'ldsUsage:\s*(\d+)', content)
                if lds_match:
                    tuning_data.LDS_allocated = int(lds_match.group(1))

                # Look for SGPR count
                sgpr_match = re.search(r'\.sgpr_count:\s+(\d+)', content)
                if sgpr_match:
                    tuning_data.sgpr_count = int(sgpr_match.group(1))
                
                # Look for VGPR count
                vgpr_match = re.search(r'\.vgpr_count:\s+(\d+)', content)
                if vgpr_match:
                    tuning_data.vgpr_count = int(vgpr_match.group(1))
                
                # Look for SGPR spill count
                sgpr_spill_match = re.search(r'\.sgpr_spill_count:\s+(\d+)',
                                             content)
                if sgpr_spill_match:
                    tuning_data.sgpr_spills = int(sgpr_spill_match.group(1))
                
                # Look for VGPR spill count
                vgpr_spill_match = re.search(r'\.vgpr_spill_count:\s+(\d+)',
                                             content)
                if vgpr_spill_match:
                    tuning_data.vgpr_spills = int(vgpr_spill_match.group(1))
                
        except Exception as e:
            print(f"Error parsing DBG file {dbg_message_file}: {e}")
    else:
        print(f"Warning: DBG file {dbg_message_file} not found")

def parse_results(gen_files):
    """
    This function parses the generated files to gather the desired information.
    gen_files will contain all of the output files from the different stages
    of compilation. It will be structured something like the following:
      - rocmlir-gen output
      - rocmlir-driver output
      - rocmlir-driver debug output
      - rocmlir-translate output
      - rocmlir-opt output
      - rocmlir-llc output  
    """
    tuning_data = create_tuning_data()

    dbg_message_file = gen_files[2]
    parse_driver_debug_results(tuning_data, dbg_message_file)

    return tuning_data
    
def parse_perf_config(perf_config, num_cu):
    """
    Parse the perfConfig string to extract tuning parameters.
    
    Format: attn:v1:MPerBlock,NPerBlock,KPerBlock,MPerWave,NPerWave,kPack,
            splitKFactor,forceUnroll,ThreadCopyMore
    
    Returns:
        dict: Dictionary containing parsed parameters
    """
    try:
        # Split by ':' to separate operation, version, and parameters
        parts = perf_config.split(':')
        if len(parts) < 2:
            raise ValueError(f"Invalid perfConfig format: {perf_config}")
        
        # For attention ops: operation:version:parameters
        # For gemm/conv ops: version:parameters
        if len(parts) >= 3:
            # Attention format - extract the parameters part (everything after the second ':')
            params_str = parts[2]
        else:
            # GEMM/conv format - parameters are after the first ':'
            params_str = parts[1]
        
        # Split parameters by comma
        params = params_str.split(',')
        if len(params) < 7:  # At minimum we need 7 parameters for splitKFactor
            raise ValueError(f"Insufficient parameters in perfConfig")
        
        # Parse the required parameters
        parsed_params = {
            'MPerBlock': int(params[0]),
            'NPerBlock': int(params[1]),
            'KPerBlock': int(params[2]),
            'MPerWave': int(params[3]),
            'NPerWave': int(params[4]),
            'kPack': int(params[5]),
            'splitKFactor': int(params[6])
        }
        
        # Calculate M*N PerWave
        parsed_params['MNPerWave'] = parsed_params['MPerWave'] * \
                                     parsed_params['NPerWave']

        # Calculate minNumWaves based on numCUs and numEUPerCU
        parsed_params['minNumWaves'] = int(num_cu) * numEUPerCU
        
        return parsed_params
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing perfConfig '{perf_config}': {e}")
        return None
    
def calculateConvN(arg_dict):
    """
    This function calculate the N value for convolution operations based on
    the provided arguments in the test vector.

    Note: Right now we are working under the assumption that we will only ever
    # need to calculate the N value for forward convolutions based on the
    # configs in tier1-tuning-data. If in the future this changes, we will need
    # to update this function to handle calculations for different types of
    # backwards convolutions.
    """
    # Forward convolution: N = batch_size * output_height * output_width
    # This is based off of the calculation that is done in TosaToLinalgNamed
    batch_size = int(arg_dict.get('-n', 1))
    input_height = int(arg_dict.get('-H', 0))
    input_width = int(arg_dict.get('-W', 0))
    pad_top = int(arg_dict.get('-p', 0))
    pad_bottom = int(arg_dict.get('-p', 0))  # Assuming symmetric padding
    pad_left = int(arg_dict.get('-q', 0))
    pad_right = int(arg_dict.get('-q', 0))   # Assuming symmetric padding
    stride_y = int(arg_dict.get('-u', 1))
    stride_x = int(arg_dict.get('-v', 1))
    filter_height = int(arg_dict.get('-y', 1))
    filter_width = int(arg_dict.get('-x', 1))
    # Assuming same dilation for both dimensions
    dilation_y = int(arg_dict.get('-l', 1))
    dilation_x = int(arg_dict.get('-j', 1))
    
    # Calculate output dimensions using the formula:
    # output_dim = ((input_dim + pad_total - 
    #               (dilation*(filter_size-1)+1)) / stride) + 1
    output_height = ((input_height + pad_top + pad_bottom - \
                     (dilation_y * (filter_height - 1) + 1)) // stride_y) + 1
    output_width = ((input_width + pad_left + pad_right - \
                     (dilation_x * (filter_width - 1) + 1)) // stride_x) + 1

    N = batch_size * output_height * output_width

    return N
    
def extract_MNG_from_config(config, operation):
    """
    Extract M, N, and G values from the testVector based on the operation type.
    
    Args:
        config: Configuration dictionary containing testVector
        operation: Operation type (e.g., 'attention', 'gemm', 'conv2d')
    
    Returns:
        tuple: (M, N, G) values based on operation type
    """
    test_vector = config["testVector"]
    test_args = parse_test_args(test_vector)
    
    # Create a dictionary of arguments for easier lookup
    arg_dict = {}
    i = 0
    while i < len(test_args):
        if test_args[i].startswith('-') and i + 1 < len(test_args):
            # Check if next arg is a value (not another flag)
            if not test_args[i + 1].startswith('-'):
                arg_dict[test_args[i]] = test_args[i + 1]
                i += 2
            else:
                # Flag without value
                arg_dict[test_args[i]] = True
                i += 1
        else:
            i += 1
    
    M = None
    N = None
    G = None
    
    try:
        if operation.lower() in ['attention', 'attn']:
            # For attention ops: M = seq_len_q, N = seq_len_k, G = g
            M = int(arg_dict.get('-seq_len_q', 0))
            N = int(arg_dict.get('-seq_len_k', 0))
            G = int(arg_dict.get('-g', 0))
            
        elif operation.lower() in ['gemm']:
            # For GEMM ops: M = m, N = n, G = g
            M = int(arg_dict.get('-m', 0))
            N = int(arg_dict.get('-n', 0))
            G = int(arg_dict.get('-g', 0))
            
        elif operation.lower() in ['conv2d', 'conv']:
            # For conv ops: M = k, N = calculateConvN, G = g
            M = int(arg_dict.get('-k', 0))
            N = calculateConvN(arg_dict)
            G = int(arg_dict.get('-g', 0))
            
        else:
            print(f"Warning: Unknown operation type '{operation}'")
            return None, None, None
            
    except (ValueError, TypeError) as e:
        print(f"Warning: Error parsing M, N, G values from testVector: {e}")
        print(f"testVector: {test_vector}")
        print(f"Parsed args: {arg_dict}")
        return None, None, None
    
    return M, N, G

def gatherOccupancyParameters(config, operation):
    '''
    This function gathers all of the parameters that are needed to calculate
    the theoretical occupancy
    '''
    perf_config = config["perfConfig (exhaustive)"]
    num_cu = config["numCUs"]
    parsed_params = parse_perf_config(perf_config, num_cu)
    
    if parsed_params is None:
        return [None] * 8  # Return None values if parsing fails
    
    # Extract the required parameters for occupancy calculation
    [M, N, G] = extract_MNG_from_config(config, operation)
    
    MPerBlock = int(parsed_params['MPerBlock'])
    NPerBlock = int(parsed_params['NPerBlock'])
    MNPerWave = int(parsed_params['MNPerWave'])
    minNumWaves = int(parsed_params['minNumWaves'])
    splitKFactor = int(parsed_params['splitKFactor'])

    return [M, N, G, MPerBlock, NPerBlock, MNPerWave, minNumWaves, splitKFactor]

def compile_and_collect_data(config, operation, binaries):
    """
    Compile and collect the resulting data points that we are interested in
    """
    # Get current timestamp in a filesystem-friendly format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compile the config
    gen_files = compile_config(config, operation, binaries, timestamp)

    # Parse the results from the compiled config
    results = parse_results(gen_files)

    # Calculate occupancy using the method in testing_metrics.py
    [M, N, G, MPerBlock, NPerBlock, MNPerWave, minNumWaves, splitKFactor] = \
                                    gatherOccupancyParameters(config, operation)
    results.occupancy = calculateOccupancy(M, N, G, MPerBlock, NPerBlock,
                                           MNPerWave, minNumWaves, splitKFactor)

    # Clean up temporary files
    for temp_file in gen_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"  Warning: Could not remove {temp_file}: {e}")

    return results

def write_results_to_csv(results, configs):
    """
    Write the collected tuning data results to a CSV file.
    
    Args:
        results: List of tuning data dictionaries
        configs: List of original configuration dictionaries
        output_file: Path to the output CSV file
    """
    if not results:
        print("No results to write")
        return
    
    # Define the fieldnames for the CSV
    fieldnames = [
        'arch',
        'numCUs',
        'testVector',
        'perfConfig',
        'blocksize',
        'gridsize',
        'vgpr_count',
        'vgpr_spills',
        'sgpr_count',
        'sgpr_spills',
        'LDS_allocated',
        'occupancy'
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tuning_results_{timestamp}.csv"
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header
            writer.writeheader()
            
            # Write each result row
            for (config, result) in zip(configs, results):
                # Convert TuningData object to a dictionary
                result_dict = result.to_dict()

                row = {
                    'arch': config.get("# arch", ""),
                    'numCUs': config.get("numCUs", ""),
                    'testVector': config.get("testVector", ""),
                    'perfConfig': config.get("perfConfig (exhaustive)", ""),
                    'blocksize': result_dict.get('blocksize', ''),
                    'gridsize': result_dict.get('gridsize', ''),
                    'vgpr_count': result_dict.get('vgpr_count', ''),
                    'vgpr_spills': result_dict.get('vgpr_spills', ''),
                    'sgpr_count': result_dict.get('sgpr_count', ''),
                    'sgpr_spills': result_dict.get('sgpr_spills', ''),
                    'LDS_allocated': result_dict.get('LDS_allocated', ''),
                    'occupancy': result_dict.get('occupancy', '')
                }
                writer.writerow(row)
        
        print(f"Results written to {output_file}")
        
    except Exception as e:
        print(f"Error writing results to CSV: {e}")

def print_progress(current, total):
    """Print a progress bar to stdout."""
    prefix = "Processing Configs"
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percent:.1f}%)', end='',
          flush=True)
    if current == total:
        print()  # New line when complete

def main():
    """Main function to process configurations and collect tuning data."""
    parser = argparse.ArgumentParser(
        description="Compile configurations and collect tuning data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--op', required=True,
                        help='Operation to perform (e.g., "compile")')
    parser.add_argument('config_csv', help='Path to the configuration CSV file')
    
    args = parser.parse_args()

    # Check to make sure that we can find the rocmlir binaries
    binaries = check_rocmlir_binaries()

    # Parse the configuration file
    configs = parse_config_csv(args.config_csv)
    print(f"Found {len(configs)} configurations to process")
    
    # Process each configuration
    results = []
    total_configs = len(configs)
    for i, config in enumerate(configs):
        print_progress(i, total_configs)
        metrics = compile_and_collect_data(config, args.op, binaries)
        results.append(metrics)
    
    #Write the results to a final CSV file
    write_results_to_csv(results, configs)

if __name__ == "__main__":
    main()
