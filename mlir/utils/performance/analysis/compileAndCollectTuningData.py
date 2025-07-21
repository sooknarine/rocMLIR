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
import os
import shutil
import subprocess
import sys
import tempfile

from datetime import datetime

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
        print("Please make sure rocMLIR is built and the build directory exists.")
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

def compile_and_collect_data(config, operation, binaries):
    """
    Compile and collect the resulting data points that we are interested in
    """
    results = []
    arch = config["# arch"].split(':')[0]

    # Get current timestamp in a filesystem-friendly format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build the rocmlir-gen command
    rocmlir_gen_cmd = [
        binaries[0],
        "--operation", operation,
        "--arch", arch,
        "--num_cu", config["numCUs"],
    ]

    # Parse and add the test vector arguments
    test_vector = config["testVector"]
    test_args = parse_test_args(test_vector)
    rocmlir_gen_cmd.extend(test_args)

    # Add perf_config
    rocmlir_gen_cmd.extend(["--perf_config",
                            config["perfConfig (exhaustive)"]])
    
    # Add output file
    rocmlir_gen_cmd.extend(["-o", f"rocmlir-gen-output-{arch}-{timestamp}.mlir"])

    # Build the rocmlir-driver command
    rocmlir_driver_cmd = [
        binaries[1],
        "-kernel-pipeline=gpu,rocdl",
        "--arch=gfx942",
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
    
    # Execute commands sequentially
    try:
        # First process: rocmlir-gen
        gen_result = subprocess.run(
            rocmlir_gen_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Second process: rocmlir-driver
        driver_result = subprocess.run(
            rocmlir_driver_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Third process: rocmlir-translate
        translate_result = subprocess.run(
            rocmlir_translate_cmd,
            capture_output=True,
            text=True,
            check=True
        )


        # Fourth process: opt
        opt_result = subprocess.run(
            opt_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Fifth process: llc
        llc_result = subprocess.run(
            llc_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Clean up temporary files
        temp_files = [
            f"rocmlir-gen-output-{arch}-{timestamp}.mlir",
            f"rocmlir-driver-output-{arch}-{timestamp}.mlir",
            f"rocmlir-translate-output-{arch}-{timestamp}.ll",
            f"rocmlir-opt-output-{arch}-{timestamp}.bc"
            f"rocmlir-opt-output-{arch}-{timestamp}.s"
        ]
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"  Warning: Could not remove {temp_file}: {e}")

    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None
    return results

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
    for config in configs:
        metrics = compile_and_collect_data(config, args.op, binaries)
        results.append(metrics)
        # TODO: Early return for debugging purposes (can remove once we get it
        # working for the first case)
        return
    
    #TODO: Need to add writing the resulting csv to the results dir
    

if __name__ == "__main__":
    main()