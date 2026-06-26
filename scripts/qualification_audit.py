import os
import numpy as np
import argparse
import struct

def audit_binary_file(filepath, dtype='f32', sentinel=1e35):
    print(f"--- Auditing: {os.path.basename(filepath)} ---")
    
    # Map dtype string to numpy dtype
    np_dtype = np.float32 if dtype == 'f32' else np.float64
    
    # Load data
    try:
        data = np.fromfile(filepath, dtype=np_dtype)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    total_elements = data.size
    print(f"Total elements: {total_elements:,}")

    # Filter sentinels
    valid_mask = np.abs(data) < sentinel
    valid_data = data[valid_mask]
    sentinel_count = total_elements - valid_data.size
    
    if sentinel_count > 0:
        print(f"Sentinels found: {sentinel_count:,} ({sentinel_count/total_elements:.2%})")
    
    if valid_data.size == 0:
        print("Error: No valid data found after sentinel filtering.")
        return

    # Statistics
    v_min, v_max = valid_data.min(), valid_data.max()
    v_mean = valid_data.mean()
    print(f"Range: [{v_min:.4e}, {v_max:.4e}]")
    print(f"Mean: {v_mean:.4e}")

    # Quantiles for selectivity selection
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    q_values = np.quantile(valid_data, quantiles)
    print("\nSelectivity Thresholds:")
    for q, v in zip(quantiles, q_values):
        print(f"  {q*100:2.0f}%: {v:.4e}")

    # Raw IEEE Leading-Byte Truncation Proxy
    # This is not BUFF/v2 Q/D/U and not a replacement for encoded-artifact qualification.
    # It acts as a rough screening proxy to evaluate significance decay.
    print("\nRaw IEEE Leading-Byte Truncation Proxy:")
    
    if dtype == 'f32':
        # F32 is 4 bytes. 
        # Byte 0: Sign + Exponent high
        # Byte 1: Exponent low + Mantissa high
        # Byte 2-3: Mantissa
        bytes_data = valid_data.view(np.uint32)
        total_depth = 4
    else:
        # F64 is 8 bytes
        bytes_data = valid_data.view(np.uint64)
        total_depth = 8

    for k in range(1, total_depth + 1):
        # Shift and mask to simulate leading k bytes
        shift = (total_depth - k) * 8
        mask = ( (1 << (k * 8)) - 1) << shift
        capped_bytes = bytes_data & mask
        
        # Cast back to float for error calculation
        if dtype == 'f32':
            capped_floats = capped_bytes.view(np.float32)
        else:
            capped_floats = capped_bytes.view(np.float64)
            
        abs_errors = np.abs(valid_data - capped_floats)
        max_error = abs_errors.max()
        avg_error = abs_errors.mean()
        
        # Show empirical max error as a proxy for the bound.
        print(f"  k={k}: Max Error = {max_error:.4e}, Avg Error = {avg_error:.4e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scientific Data Qualification Audit")
    parser.add_argument("--file", required=True, help="Path to binary file")
    parser.add_argument("--dtype", choices=['f32', 'f64'], default='f32', help="Data type (f32 or f64)")
    parser.add_argument("--sentinel", type=float, default=1e35, help="Sentinel value threshold")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
    else:
        audit_binary_file(args.file, args.dtype, args.sentinel)
