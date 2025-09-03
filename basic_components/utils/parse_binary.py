# Parse binary datasets into NumPy arrays

import numpy as np
import struct # For reading binary data

def parse_idx_images(filepath):
    # used by basic_NN_multi-class-classification.ipynb
    """Parses an IDX image file (.idx3-ubyte) into a NumPy array."""
    with open(filepath, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        # ">IIII" means:
        # > : big-endian byte order
        # I : unsigned int (4 bytes)
        # So, 4 unsigned ints are read, each 4 bytes long (total 16 bytes for header)

        if magic != 2051: # Expected magic number for image files
            raise ValueError(f"Magic number mismatch: Expected 2051, got {magic}")

        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images

def parse_idx_labels(filepath):
    # Used by basic_NN_multi-class-classification.ipynb
    """Parses an IDX label file (.idx1-ubyte) into a NumPy array."""
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        # ">II" means:
        # > : big-endian byte order
        # I : unsigned int (4 bytes)
        # So, 2 unsigned ints are read (total 8 bytes for header)

        if magic != 2049: # Expected magic number for label files
            raise ValueError(f"Magic number mismatch: Expected 2049, got {magic}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


