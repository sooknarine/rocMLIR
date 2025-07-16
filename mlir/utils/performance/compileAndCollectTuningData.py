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
python3 compileAndCollectTuningData.py <config.csv>
"""

import argparse
import csv
import sys

def parse_config_csv(config_file):
    """Parse the input CSV file containing configuration data."""
    configs = []
    print(config_file)
    try:
        with open(config_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                configs.append(row)
        return configs
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

def main():
    """Main function to process configurations and collect tuning data."""
    parser = argparse.ArgumentParser(
        description="Compile configurations and collect tuning data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('config_csv', help='Path to the configuration CSV file')
    
    args = parser.parse_args()
    
    # Parse the configuration file
    configs = parse_config_csv(args.config_csv)
    print(f"Found {len(configs)} configurations to process")
    
    # Process each configuration
    results = []
    """
    for i, config in enumerate(configs, 1):
        print(f"\nProcessing configuration {i}/{len(configs)}")
        metrics = compile_and_collect_data(config)
        
        # Combine config with collected metrics
        result = {**config, **metrics}
        results.append(result)
    """

    # TODO: Output results (could write to CSV, JSON, etc.)
    print(f"\nProcessed {len(results)} configurations successfully")

if __name__ == "__main__":
    main()