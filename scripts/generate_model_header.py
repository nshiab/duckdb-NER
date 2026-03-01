import sys
import os

if len(sys.argv) < 2:
    print("Usage: python generate_model_header.py model.bin [output.hpp]")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else "src/include/default_model.hpp"

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found")
    sys.exit(1)

with open(input_path, "rb") as f:
    data = f.read()

with open(output_path, "w") as f:
    f.write("#ifndef DEFAULT_MODEL_HPP
")
    f.write("#define DEFAULT_MODEL_HPP

")
    f.write("#include <stdint.h>
")
    f.write("#include <stddef.h>

")
    f.write(f"// Bundled model from {os.path.basename(input_path)}
")
    f.write(f"// Original size: {len(data)} bytes
")
    f.write("static const uint8_t DEFAULT_MODEL_DATA[] = {
")
    
    # Write bytes in chunks of 12 for readability
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        hex_chunk = ", ".join([f"0x{b:02x}" for b in chunk])
        comma = "," if i + 12 < len(data) else ""
        f.write(f"    {hex_chunk}{comma}
")
        
    f.write("};

")
    f.write("static const size_t DEFAULT_MODEL_SIZE = sizeof(DEFAULT_MODEL_DATA);

")
    f.write("#endif // DEFAULT_MODEL_HPP
")

print(f"Done! Header generated at {output_path}")
