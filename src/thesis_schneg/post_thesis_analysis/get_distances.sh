#!/bin/bash
source ../../.env
# Define the available datasets
datasets=("aql" "orcas" "aol" "ms-marco")

# Output file in the current directory
output_file="$EMBEDDINGS_PATH/embeddings-distances/swd_results.csv"

# Remove output file if it already exists
[ -f "$output_file" ] && rm "$output_file"

# Write CSV header
echo "dataset1,dataset2,distance" > "$output_file"

echo "Starting SWD benchmark for all pairs..."

# Outer loop for the first dataset
for i in "${!datasets[@]}"; do
    for j in "${!datasets[@]}"; do
        
        # Avoid comparing the same dataset (e.g., aq vs aq)
        # and avoid redundant pairs (e.g., if aq-orc is done, skip orc-aq)
        if [ "$i" -lt "$j" ]; then
            ds1=${datasets[$i]}
            ds2=${datasets[$j]}
            
            echo "------------------------------------------------"
            echo "Calculating: $ds1 vs $ds2"
            
            # Execute your CLI command
            # We assume the command outputs the float distance. 
            # If your command outputs more text, we might need to parse it.
            result=$(thesis-schneg embeddings-analysis \
                --datasets "$ds1" "$ds2" \
                --analysis embeddings-distance \
                --device-type gpu \
                --batch-size 100 \
                --num-input-files 15)
            
            # Simple duration measurement in Bash
            # (Though your CLI already returns duration, this is a fallback)
            distance=$(echo "$result" | tail -n 1)
            echo "Result: $distance"
            
            # Append to CSV
            echo "$ds1,$ds2,$distance" >> "$output_file"
        fi
    done
done

echo "------------------------------------------------"
echo "Benchmark finished. Results saved in $output_file"