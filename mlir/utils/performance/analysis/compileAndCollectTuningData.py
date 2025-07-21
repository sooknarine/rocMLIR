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

# Global template for tuning data structure
TUNING_DATA_TEMPLATE = {
    'blocksize': None,
    'gridsize': None,
    'vgpr_count': None,
    'vgpr_spills': None,
    'sgpr_count': None,
    'sgpr_spills': None,
    'LDS_allocated': None,
    'occupancy': None
}

def create_tuning_data():
    return TUNING_DATA_TEMPLATE.copy()

def check_rocmlir_binaries():
    """
    Check if rocmlir-gen and rocmlir-driver binaries can be found.
    Returns the paths to the binaries if found, otherwise exits.
    """
    # Get the script's directory and find the rocMLIR root
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up from mlir/utils/performance to find rocMLIR root
    rocmlir_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..',
                                                '..'))

    # Define expected binary paths
    build_bin_dir = os.path.join(rocmlir_root, 'build', 'bin')
    build_external_bin_dir = os.path.join(rocmlir_root, 'build', 'external',
                                          'llvm-project', 'llvm', 'bin')
    rocmlir_gen_path = os.path.join(build_bin_dir, 'rocmlir-gen')
    rocmlir_driver_path = os.path.join(build_bin_dir, 'rocmlir-driver')
    rocmlir_translate_path = os.path.join(build_bin_dir, 'rocmlir-translate')
    opt_path = os.path.join(build_external_bin_dir, 'opt')
    llc_path = os.path.join(build_external_bin_dir, 'llc')

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

    # Check for rocmlir-translate
    if not os.path.exists(rocmlir_translate_path):
        print(f"Error: rocmlir-translate not found at {rocmlir_translate_path}")
        sys.exit(1)

    # Check for opt
    if not os.path.exists(opt_path):
        print(f"Error: opt not found at {opt_path}")
        sys.exit(1)

    # Check for llc
    if not os.path.exists(llc_path):
        print(f"Error: llc not found at {llc_path}")
        sys.exit(1)

    print(f"✓ Found rocmlir-gen: {rocmlir_gen_path}")
    print(f"✓ Found rocmlir-driver: {rocmlir_driver_path}")
    print(f"✓ Found rocmlir-translate: {rocmlir_translate_path}")
    print(f"✓ Found opt: {opt_path}")
    print(f"✓ Found llc: {llc_path}")

    return [rocmlir_gen_path, rocmlir_driver_path, rocmlir_translate_path,
           opt_path, llc_path]

def parse_config_csv(config_file):
    """Parse the input CSV file containing configuration data."""
    configs = []

    try:
        with open(config_file, 'r') as csvfile:
            # Read the content and process manually
            lines = csvfile.readlines()
            
            if not lines:
                return configs
            
            # Parse header
            header_line = lines[0].strip()
            headers = [h.strip() for h in header_line.split('\t')]
            
            # Process each data line
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                
                # Split by tabs (not commas, since commas are within the
                # perfConfig field)
                values = line.split('\t')
                
                # Create dictionary for this row
                clean_row = {}
                for i, header in enumerate(headers):
                    if i < len(values):
                        value = values[i].strip()
                        clean_row[header] = value
                    else:
                        clean_row[header] = ""
                
                if clean_row:  # Only add non-empty rows
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

def compile_config(config, operation, binaries, timestamp):
    arch = config["# arch"].split(':')[0]
    num_cu = config["numCUs"]

    # Build the rocmlir-gen command
    rocmlir_gen_cmd = [
        binaries[0],
        "--operation", operation,
        "--arch", arch,
        "--num_cu", num_cu,
    ]

    # Parse and add the test vector arguments
    test_vector = config["testVector"]
    test_args = parse_test_args(test_vector)
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
        "-kernel-pipeline=gpu,rocdl",
        "--arch=gfx942",
        "--debug-only=convert-rock-to-gpu",
        f"rocmlir-gen-output-{arch}-{timestamp}.mlir",
        "-o", f"rocmlir-driver-output-{arch}-{timestamp}.mlir"
    ]

    # Build the rocmlir-translate command
    rocmlir_translate_cmd = [
        binaries[2],
        "-gpu-module-to-rocdlir",
        f"rocmlir-driver-output-{arch}-{timestamp}.mlir",
        "-o", f"rocmlir-translate-output-{arch}-{timestamp}.ll"
    ]

    # Build the opt command
    opt_cmd = [
        binaries[3],
        "-O3",
        f"rocmlir-translate-output-{arch}-{timestamp}.ll",
        "-o", f"rocmlir-opt-output-{arch}-{timestamp}.bc"
    ]

    # Build the llc command
    llc_cmd = [
        binaries[4],
        f"-mcpu={arch}",
        f"rocmlir-opt-output-{arch}-{timestamp}.bc"
    ]

    commands = [rocmlir_gen_cmd, rocmlir_driver_cmd, rocmlir_translate_cmd,
                opt_cmd, llc_cmd]
    
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
        print(f"Command failed: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None
    
    return [
        f"rocmlir-gen-output-{arch}-{timestamp}.mlir",
        f"rocmlir-driver-output-{arch}-{timestamp}.mlir",
        f"rocmlir-driver-debug-{arch}-{timestamp}.txt",
        f"rocmlir-translate-output-{arch}-{timestamp}.ll",
        f"rocmlir-opt-output-{arch}-{timestamp}.bc",
        f"rocmlir-opt-output-{arch}-{timestamp}.s"
    ]

def parse_llc_results(tuning_data, llc_file):
    # Parse the LLC assembly file for SGPR and VGPR information
    if os.path.exists(llc_file):
        try:
            with open(llc_file, 'r') as f:
                content = f.read()
                
                # Look for SGPR count
                sgpr_match = re.search(r'\.sgpr_count:\s+(\d+)', content)
                if sgpr_match:
                    tuning_data['sgpr_count'] = int(sgpr_match.group(1))
                
                # Look for VGPR count
                vgpr_match = re.search(r'\.vgpr_count:\s+(\d+)', content)
                if vgpr_match:
                    tuning_data['vgpr_count'] = int(vgpr_match.group(1))
                
                # Look for SGPR spill count
                sgpr_spill_match = re.search(r'\.sgpr_spill_count:\s+(\d+)',
                                             content)
                if sgpr_spill_match:
                    tuning_data['sgpr_spills'] = int(sgpr_spill_match.group(1))
                
                # Look for VGPR spill count
                vgpr_spill_match = re.search(r'\.vgpr_spill_count:\s+(\d+)',
                                             content)
                if vgpr_spill_match:
                    tuning_data['vgpr_spills'] = int(vgpr_spill_match.group(1))
                
        except Exception as e:
            print(f"Error parsing LLC file {llc_file}: {e}")
    else:
        print(f"Warning: LLC file {llc_file} not found")

def parse_driver_debug_results(tuning_data, dbg_message_file):
    # Parse the rocmlir-driver debug output file for gridsize, blocksize, and
    # lds usage information
    if os.path.exists(dbg_message_file):
        try:
            with open(dbg_message_file, 'r') as f:
                content = f.read()
                
                # Look for blocksize
                blocksize_match = re.search(r'blockSize:\s*(\d+)', content)
                if blocksize_match:
                    tuning_data['blocksize'] = int(blocksize_match.group(1))

                # Look for gridsize
                gridsize_match = re.search(r'gridSize:\s*(\d+)', content)
                if gridsize_match:
                    tuning_data['gridsize'] = int(gridsize_match.group(1))

                # Look for LDS_allocated
                lds_match = re.search(r'ldsUsage:\s*(\d+)', content)
                if lds_match:
                    tuning_data['LDS_allocated'] = int(lds_match.group(1))
                
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

    llc_file = gen_files[-1]
    parse_llc_results(tuning_data, llc_file)

    dbg_message_file = gen_files[2]
    parse_driver_debug_results(tuning_data, dbg_message_file)

    print(tuning_data)
    return tuning_data
    

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
            for i, (config, result) in enumerate(zip(configs, results)):
                row = {
                    'arch': config.get("# arch", ""),
                    'numCUs': config.get("numCUs", ""),
                    'testVector': config.get("testVector", ""),
                    'perfConfig': config.get("perfConfig (exhaustive)", ""),
                    'blocksize': result.get('blocksize', ''),
                    'gridsize': result.get('gridsize', ''),
                    'vgpr_count': result.get('vgpr_count', ''),
                    'vgpr_spills': result.get('vgpr_spills', ''),
                    'sgpr_count': result.get('sgpr_count', ''),
                    'sgpr_spills': result.get('sgpr_spills', ''),
                    'LDS_allocated': result.get('LDS_allocated', ''),
                    'occupancy': result.get('occupancy', '')
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
        # TODO: Early return for debugging purposes (can remove once we get it
        # working for the first case)
        break
    
    #Write the results to a final CSV file
    write_results_to_csv(results, configs)

if __name__ == "__main__":
    main()