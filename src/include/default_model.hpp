#ifndef DEFAULT_MODEL_HPP
#define DEFAULT_MODEL_HPP

#include <stdint.h>
#include <stddef.h>

// This is a placeholder for the bundled tiny NER model.
// In a real scenario, this would be populated with the bytes of a quantized bert-tiny-ner.bin.
// Size: ~4.5MB for Q4_0 quantized bert-tiny.
static const uint8_t DEFAULT_MODEL_DATA[] = {
    0x67, 0x67, 0x6d, 0x6c, // magic
    0x00, 0x00, 0x00, 0x00  // dummy data...
};

static const size_t DEFAULT_MODEL_SIZE = sizeof(DEFAULT_MODEL_DATA);

#endif // DEFAULT_MODEL_HPP
