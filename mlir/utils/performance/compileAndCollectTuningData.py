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

def check_rocmlir_binaries():
    """
    Check if rocmlir-gen and rocmlir-driver binaries can be found.
    Returns the paths to the binaries if found, otherwise exits.
    """
    # Get the script's directory and find the rocMLIR root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up from mlir/utils/performance to find rocMLIR root
    rocmlir_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    
    # Define expected binary paths
    build_bin_dir = os.path.join(rocmlir_root, 'build', 'bin')
    rocmlir_gen_path = os.path.join(build_bin_dir, 'rocmlir-gen')
    rocmlir_driver_path = os.path.join(build_bin_dir, 'rocmlir-driver')
    
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
    
    print(f"✓ Found rocmlir-gen: {rocmlir_gen_path}")
    print(f"✓ Found rocmlir-driver: {rocmlir_driver_path}")
    
    return rocmlir_gen_path, rocmlir_driver_path

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

def compile_and_collect_data(config, operation):
    """
    Compile and collect the resulting data points that we are interested in
    """

    # TODO: Need to add in a function to parse the testVector so that we remove
    # the flags that do not need to be applied. I also need to add in some logic
    # for making sure that we can find the proper binaries.
    """
    # Build the rocmlir-gen command
    rocmlir_gen_cmd = [
        "~/rocMLIR/build/bin/rocmlir-gen",
        "--operation", "attention",
        "--arch", config["# arch"],
        "--num_cu", config["numCUs"]
    ]

    # Parse and add the test vector arguments
    test_vector = config["testVector"]
    test_args = test_vector.split()
    rocmlir_gen_cmd.extend(test_args)

    # Add perf_config
    rocmlir_gen_cmd.extend(["--perf_config", config["perfConfig (exhaustive)"]])
    
    # Build the rocmlir-driver command
    rocmlir_driver_cmd = [
        "~/rocMLIR/build/bin/rocmlir-driver",
        "-c",
        "--debug-only=serialize-to-blob"
    ]

    try:
        # Execute the piped command
        # First process: rocmlir-gen
        gen_process = subprocess.Popen(
            rocmlir_gen_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False
        )
        
        # Second process: rocmlir-driver (takes input from gen_process)
        driver_process = subprocess.Popen(
            rocmlir_driver_cmd,
            stdin=gen_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr and stdout
            text=True,
            shell=False
        )
        
        # Close the stdout of the first process to allow it to terminate
        gen_process.stdout.close()
        
        # Wait for both processes to complete and get output
        driver_output, _ = driver_process.communicate()
        gen_process.wait()
        
        # Write output to file
        with open("compile-output.mlir", "w") as f:
            f.write(driver_output)
        
        # Parse the output to extract metrics
        metrics = parse_compile_output(driver_output)
        
        return metrics
        
    except Exception as e:
        print(f"Error executing compilation command: {e}")
        return None
    """

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
    rocmlir_gen_path, rocmlir_driver_path = check_rocmlir_binaries()

    # Parse the configuration file
    configs = parse_config_csv(args.config_csv)
    print(f"Found {len(configs)} configurations to process")
    
    # Process each configuration
    results = []
    for config in enumerate(configs):
        metrics = compile_and_collect_data(config, args.op)
        results.append(metrics)

if __name__ == "__main__":
    main()