"""
This script compiles a given config in order to collect the following data
points for each config:
- Blocksize
- Gridsize
- vgpr
- sgpr
- LDS allocated
- Occupancy

The given config is expected to be a tsv with the following format:
|# arch| numCUs | testVector | perfConfig (exhaustive) |

Usage:
    python3 compileAndCollectTuningData.py --op <operation> <config.tsv>
"""

import argparse
import csv
import os
import perfRunner
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
        self.WFsPerWG = None
        self.mfma_wmma_instruction = None
    
    def to_dict(self):
        """Convert to dictionary format for tsv writing."""
        return {
            'blocksize': self.blocksize,
            'gridsize': self.gridsize,
            'vgpr_count': self.vgpr_count,
            'vgpr_spills': self.vgpr_spills,
            'sgpr_count': self.sgpr_count,
            'sgpr_spills': self.sgpr_spills,
            'LDS_allocated': self.LDS_allocated,
            'occupancy': self.occupancy,
            'WFsPerWG' : self.WFsPerWG,
            'mfma_wmma_instruction': self.mfma_wmma_instruction
        }

def create_tuning_data():
    return TuningData()

def get_perf_config(operation, test_vector, arch, num_cu):
    """
    Get the performance configuration for the given test vector, architecture,
    and number of compute units.
    
    Args:
        test_vector: The test vector string.
        arch: The architecture string.
        num_cu: The number of compute units.
    
    Returns:
        str: The performance configuration string.
    """
    conf_class = perfRunner.PerfConfiguration
    if (operation == 'attention'):
        conf_class = perfRunner.AttentionConfiguration.fromCommandLine(test_vector.split(sep=' '), arch, num_cu)
    elif (operation == 'gemm'):
        conf_class = perfRunner.GemmConfiguration.fromCommandLine(test_vector.split(sep=' '), arch, num_cu)
    elif (operation == 'conv'):
        conf_class = perfRunner.ConvConfiguration.fromCommandLine(test_vector.split(sep=' '), arch, num_cu)

    return conf_class

def compile_config(config, perf_config, operation, paths, timestamp):
    arch = config[0].split(':')[0]
    num_cu = config[1]
    test_vector = config[2]

    conf_class = get_perf_config(operation, test_vector, arch, num_cu)
    conf_class.setPerfConfig(perf_config)
    rocmlir_gen_options = conf_class.generateMlirDriverCommandLine("")

    # Build the rocmlir-gen command
    rocmlir_gen_cmd = [paths.mlir_paths.rocmlir_gen_path] + rocmlir_gen_options.split()

    # Build the rocmlir-driver command
    rocmlir_driver_cmd = [
        paths.mlir_paths.rocmlir_driver_path,
        "-c",
        f"--arch={arch}",
        "--debug-only=convert-rock-to-gpu,serialize-to-isa",
    ]

    commands = [rocmlir_gen_cmd, rocmlir_driver_cmd]
    out, err = perfRunner.runPipeline(commands)

    # Write debug output to file
    debug_file = f"rocmlir-driver-debug-{arch}-{timestamp}.txt"
    with open(debug_file, 'w') as f:
        if isinstance(err, bytes):
            f.write(err.decode('utf-8'))
        else:
            f.write(err if err else "")
    
    return f"rocmlir-driver-debug-{arch}-{timestamp}.txt"

def parse_mfma_wmma_instructions(content):
    """
    Parse MFMA and WMMA instructions from the debug output.
    
    Args:
        content: String content of the debug output file
        
    Returns:
        list: Unique list of MFMA/WMMA instruction names
    """
    # Pattern to match MFMA and WMMA instructions
    full_pattern = r'\b(v_(?:mfma|wmma)_[a-zA-Z0-9_]+)\b'
    full_matches = re.findall(full_pattern, content, re.IGNORECASE)
    
    # Remove duplicates and sort for consistent output
    unique_instructions = sorted(list(set(full_matches)))
    
    # Assert that there is only one unique instruction
    size = len(unique_instructions)
    assert size == 1, \
           f"Expected exactly one unique MFMA/WMMA instruction, found: {size}"
    
    return unique_instructions

def parse_driver_debug_results(tuning_data, dbg_message_file):
    # Parse the rocmlir-driver debug output file for gridsize, blocksize,
    # lds usage, SGPR, and VGPR information
    if os.path.exists(dbg_message_file):
        try:
            with open(dbg_message_file, 'r') as f:
                content = f.read()
                # Look for blocksize
                blocksize_match = re.search(r'blockSize:\s*(\d+)', content)
                if not blocksize_match:
                    raise ValueError(f"Could not find blockSize in output")
                tuning_data.blocksize = int(blocksize_match.group(1))

                # Look for gridsize
                gridsize_match = re.search(r'gridSize:\s*(\d+)', content)
                if not gridsize_match:
                    raise ValueError(f"Could not find gridSize in output")
                tuning_data.gridsize = int(gridsize_match.group(1))

                # Look for waveSize
                wavesize_match = re.search(r'waveSize:\s*(\d+)', content)
                if not wavesize_match:
                    raise ValueError(f"Could not find waveSize in output")
                tuning_data.WFsPerWG = int(blocksize_match.group(1)) / int(wavesize_match.group(1))

                # Look for LDS_allocated
                lds_match = re.search(r'ldsUsage:\s*(\d+)', content)
                if not lds_match:
                    raise ValueError(f"Could not find ldsUsage in output")
                tuning_data.LDS_allocated = int(lds_match.group(1))

                # Look for SGPR count
                sgpr_match = re.search(r'\.sgpr_count:\s+(\d+)', content)
                if not sgpr_match:
                    raise ValueError(f"Could not find sgpr_count in output")
                tuning_data.sgpr_count = int(sgpr_match.group(1))
                
                # Look for VGPR count
                vgpr_match = re.search(r'\.vgpr_count:\s+(\d+)', content)
                if not vgpr_match:
                    raise ValueError(f"Could not find vgpr_count in output")
                tuning_data.vgpr_count = int(vgpr_match.group(1))
                
                # Look for SGPR spill count
                sgpr_spill_match = re.search(r'\.sgpr_spill_count:\s+(\d+)',
                                             content)
                if not sgpr_spill_match:
                    raise ValueError(f"Could not find sgpr_spill_count in output")
                tuning_data.sgpr_spills = int(sgpr_spill_match.group(1))
                
                # Look for VGPR spill count
                vgpr_spill_match = re.search(r'\.vgpr_spill_count:\s+(\d+)',
                                             content)
                if not vgpr_spill_match:
                    raise ValueError(f"Could not find vgpr_spill_count in output")
                tuning_data.vgpr_spills = int(vgpr_spill_match.group(1))

                mfma_wmma_instructions = parse_mfma_wmma_instructions(content)
                tuning_data.mfma_wmma_instruction = mfma_wmma_instructions[0]
                
        except Exception as e:
            print(f"Error parsing DBG file {dbg_message_file}: {e}")
    else:
        print(f"Warning: DBG file {dbg_message_file} not found")

def parse_results(debug_output):
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
    parse_driver_debug_results(tuning_data, debug_output)

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
            # Attention format - extract the parameters part (everything after
            # the second ':')
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
    This function calculate the total number of output elements/pixels
    for convolution operations based on the provided arguments in the test
    vector.

    Note: Right now we are working under the assumption that we will only ever
    # need to calculate the N value for forward convolutions based on the
    # configs in tier1-tuning-data. If in the future this changes, we will need
    # to update this function to handle calculations for different types of
    # backwards convolutions.
    """
    assert int(arg_dict.get('-F')) == 1, \
           "Only forward convolution (-F=1) is supported"
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
    
def extract_MNG_from_config(config, test_args, operation):
    """
    Extract M, N, and G values from the testVector based on the operation type.
    
    Args:
        config: Configuration dictionary containing testVector
        test_args: Filtered list of arguments from the testVector
        operation: Operation type (e.g., 'attention', 'gemm', 'conv2d')
    
    Returns:
        tuple: (M, N, G) values based on operation type
    """
    args = test_args.split()
    # Create a dictionary of arguments for easier lookup
    arg_dict = {}
    i = 0
    while i < len(args):
        if args[i].startswith('-') and i + 1 < len(args):
            # Check if next arg is a value (not another flag)
            if not args[i + 1].startswith('-'):
                arg_dict[args[i]] = args[i + 1]
                i += 2
            else:
                # Flag without value
                arg_dict[args[i]] = True
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
            G = int(arg_dict.get('-g', 0)) * int(arg_dict.get('-num_heads_q', 0)) 
            
        elif operation.lower() in ['gemm']:
            # For GEMM ops: M = m, N = n, G = g
            M = int(arg_dict.get('-m', 0))
            N = int(arg_dict.get('-n', 0))
            G = int(arg_dict.get('-g', 0))
            
        elif operation.lower() in ['conv', 'convfp16', 'convbfp16', 'convint8',
                                   'convfp8']:
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

def gatherOccupancyParameters(config, perf_config, test_args, operation):
    '''
    This function gathers all of the parameters that are needed to calculate
    the theoretical occupancy
    '''
    num_cu = config[1]
    parsed_params = parse_perf_config(perf_config, num_cu)
    
    if parsed_params is None:
        return [None] * 8  # Return None values if parsing fails
    
    # Extract the required parameters for occupancy calculation
    [M, N, G] = extract_MNG_from_config(config, test_args, operation)
    
    MPerBlock = int(parsed_params['MPerBlock'])
    NPerBlock = int(parsed_params['NPerBlock'])
    MNPerWave = int(parsed_params['MNPerWave'])
    minNumWaves = int(parsed_params['minNumWaves'])
    splitKFactor = int(parsed_params['splitKFactor'])

    return [M, N, G, MPerBlock, NPerBlock, MNPerWave, minNumWaves, splitKFactor]

def compile_and_collect_data(config, perf_config, test_args, operation,
                             binaries):
    """
    Compile and collect the resulting data points that we are interested in
    """
    # Get current timestamp in a filesystem-friendly format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compile the config
    debug_output = compile_config(config, perf_config, operation, binaries,
                                  timestamp)

    # Parse the results from the compiled config
    results = parse_results(debug_output)

    # Calculate occupancy using the method in testing_metrics.py
    [M, N, G, MPerBlock, NPerBlock,
        MNPerWave, minNumWaves, splitKFactor] = \
                                gatherOccupancyParameters(config,
                                                          perf_config,
                                                          test_args,
                                                          operation)
    results.occupancy = calculateOccupancy(M, N, G, MPerBlock, NPerBlock,
                                            MNPerWave, minNumWaves,
                                            splitKFactor)

    # Clean up temporary debug file
    try:
        if os.path.exists(debug_output):
            os.remove(debug_output)
    except Exception as e:
        print(f"  Warning: Could not remove {debug_output}: {e}")

    return results

def write_results_to_tsv(results, configs):
    """
    Write the collected tuning data results to a tsv file.
    
    Args:
        results: List of tuning data dictionaries
        configs: List of original configuration dictionaries
        output_file: Path to the output tsv file
    """
    if not results:
        print("No results to write")
        return
    
    # Define the fieldnames for the tsv
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
        'occupancy',
        'WFsPerWG',
        'mfma_wmma_instruction'
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tuning_results_{timestamp}.tsv"
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames)
            
            # Write the header
            writer.writeheader()
            
            # Write each result row
            for (config, result) in zip(configs, results):
                arch, num_cu, test_vector = config
                if result is None:
                    row = {
                        'arch': arch,
                        'numCUs': num_cu,
                        'testVector': test_vector,
                        'perfConfig': configs[config],
                        'blocksize': None,
                        'gridsize': None,
                        'vgpr_count': None,
                        'vgpr_spills': None,
                        'sgpr_count': None,
                        'sgpr_spills': None,
                        'LDS_allocated': None,
                        'occupancy': None,
                        'WFsPerWG': None,
                        'mfma_wmma_instruction': None
                    }
                else:
                    # Convert TuningData object to a dictionary
                    result_dict = result.to_dict()

                    row = {
                        'arch': arch,
                        'numCUs': num_cu,
                        'testVector': test_vector,
                        'perfConfig': configs[config],
                        'blocksize': result_dict.get('blocksize', ''),
                        'gridsize': result_dict.get('gridsize', ''),
                        'vgpr_count': result_dict.get('vgpr_count', ''),
                        'vgpr_spills': result_dict.get('vgpr_spills', ''),
                        'sgpr_count': result_dict.get('sgpr_count', ''),
                        'sgpr_spills': result_dict.get('sgpr_spills', ''),
                        'LDS_allocated': result_dict.get('LDS_allocated', ''),
                        'occupancy': result_dict.get('occupancy', ''),
                        'WFsPerWG': result_dict.get('WFsPerWG', ''),
                        'mfma_wmma_instruction': result_dict.get('mfma_wmma_instruction', '')
                    }
                
                writer.writerow(row)
        
        print(f"\nResults written to {output_file}")
        
    except Exception as e:
        print(f"\nError writing results to tsv: {e}")

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
                        help='Operation to perform (e.g., "compile")',
                        choices=['conv', 'gemm', 'attention'])
    parser.add_argument('config_tsv', help='Path to the tuning database file')
    
    args = parser.parse_args()

    # Get the paths to the rocmlir binaries
    build_bin_dir = os.path.dirname(os.path.abspath(__file__))
    rocmlir_root = os.path.dirname(build_bin_dir)
    paths = perfRunner.create_paths(None, rocmlir_root)

    # Parse the configuration file
    configs = perfRunner.read_tuning_db(args.config_tsv, True)
    op_configs = None
    if (args.op == 'conv'):
        op_configs = perfRunner.getConvConfigurations(args.config_tsv)
    elif (args.op == 'gemm'):
        op_configs = perfRunner.getGemmConfigurations(args.config_tsv)
    elif (args.op == 'attention'):
        op_configs = perfRunner.getAttentionConfigurations(args.config_tsv)
    else:
        print(f"Error: Unknown operation '{args.op}'")
        sys.exit(1)
    print(f"Found {len(configs)} configurations to process")
    
    # Process each configuration
    results = []
    total_configs = len(configs)
    for i, (config, test_args) in enumerate(zip(configs, op_configs)):
        print_progress(i, total_configs)
        metrics = compile_and_collect_data(config, configs[config], test_args,
                                           args.op, paths)
        results.append(metrics)

    #Write the results to a final tsv file
    write_results_to_tsv(results, configs)

if __name__ == "__main__":
    main()
