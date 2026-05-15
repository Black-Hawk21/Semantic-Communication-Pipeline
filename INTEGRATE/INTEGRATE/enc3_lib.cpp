/*
 * enc3_lib.cpp — Refactored encryption library for Python/pybind11 integration
 *
 * Changes from original enc3.cpp:
 * 1. Removed main() and all stdin/stdout I/O
 * 2. Exposed encrypt_data() / decrypt_data() as the public API
 * 3. Added batch processing (encrypt_chunks / decrypt_chunks)
 * 4. Made padding deterministic (seeded RNG) so encrypt is reproducible
 * 5. Added pybind11 module at the bottom
 * 6. CHANGED: Padding now occurs after EVERY SINGLE data bit.
 * 7. CHANGED: Padding length is now 8, 9, or 10 bits.
 *
 * Build (standalone test):
 * g++ -O2 -std=c++17 -o enc3_test enc3_lib.cpp -DSTANDALONE_TEST
 *
 * Build (Python module):
 * c++ -O3 -shared -std=c++17 -fPIC \
 * $(python3 -m pybind11 --includes) \
 * enc3_lib.cpp -o enc3$(python3-config --extension-suffix)
 */

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <sstream>

// =====================================================================
//  Types
// =====================================================================

using Bits = std::vector<int>;

// =====================================================================
//  String <-> Bits
// =====================================================================

static Bits string_to_bits(const std::string &s) {
    Bits b;
    b.reserve(s.size() * 8);
    for (char c : s)
        for (int i = 0; i < 8; i++)
            b.push_back((c >> i) & 1);
    return b;
}

static std::string bits_to_string(const Bits &b) {
    std::string s;
    s.reserve(b.size() / 8 + 1);
    for (size_t i = 0; i < b.size(); i += 8) {
        char c = 0;
        for (int j = 0; j < 8 && i + j < b.size(); j++)
            c |= (b[i + j] << j);
        s.push_back(c);
    }
    return s;
}

// =====================================================================
//  Key
// =====================================================================

struct Key {
    std::vector<uint32_t> constants;
    uint32_t mask;
};

static Key generate_key(uint32_t seed) {
    std::mt19937 rng(seed);
    Key k;
    k.constants.resize(6);
    for (int i = 0; i < 6; i++)
        k.constants[i] = rng();
    k.mask = rng();
    return k;
}

// =====================================================================
//  Bit Operations
// =====================================================================

static Bits xor_bits(Bits a, const Bits &b) {
    for (size_t i = 0; i < a.size(); i++)
        a[i] ^= b[i % b.size()];
    return a;
}

static Bits rotate_left(Bits a, int k) {
    int n = (int)a.size();
    Bits r(n);
    for (int i = 0; i < n; i++)
        r[i] = a[((i - k) % n + n) % n];
    return r;
}

static Bits rotate_right(Bits a, int k) {
    return rotate_left(a, (int)a.size() - k);
}

static Bits reverse_bits(Bits a) {
    std::reverse(a.begin(), a.end());
    return a;
}

static Bits swap_halves(Bits a) {
    size_t mid = a.size() / 2;
    Bits r;
    r.reserve(a.size());
    r.insert(r.end(), a.begin() + mid, a.end());
    r.insert(r.end(), a.begin(), a.begin() + mid);
    return r;
}

// =====================================================================
//  Hash (MAC)
// =====================================================================

static uint32_t simple_hash(const Bits &b) {
    uint32_t h = 0x9e3779b9;
    for (size_t i = 0; i < b.size(); i++)
        h ^= (b[i] + (h << 6) + (h >> 2));
    return h;
}

// =====================================================================
//  Padding — NOW DETERMINISTIC (seeded from key + chunk index)
// =====================================================================

static Bits apply_padding(Bits data, uint32_t pad_seed) {
    std::mt19937 rng(pad_seed);

    int x = rng() % 3 + 8; // CHANGED: Now gives 8, 9, or 10

    Bits result;
    result.reserve(data.size() * (1 + x));

    // Process 1 bit at a time
    for (size_t i = 0; i < data.size(); i++) {
        result.push_back(data[i]);
        for (int j = 0; j < x; j++)
            result.push_back(rng() & 1);
    }

    uint32_t mac = simple_hash(result);

    // Prepend metadata: 4 bits for x, 32 bits for MAC
    Bits meta;
    meta.reserve(36 + result.size());

    for (int i = 0; i < 4; i++) // CHANGED: 4 bits to safely store 8, 9, 10
        meta.push_back((x >> i) & 1);
    for (int i = 0; i < 32; i++)
        meta.push_back((mac >> i) & 1);

    meta.insert(meta.end(), result.begin(), result.end());
    return meta;
}

static Bits remove_padding(const Bits &data, bool &tampered) {
    if (data.size() < 36) { // CHANGED: Metadata is 36 bits long
        tampered = true;
        return {};
    }

    int x = 0;
    for (int i = 0; i < 4; i++) // CHANGED: Read 4 bits
        x |= (data[i] << i);

    uint32_t expected_mac = 0;
    for (int i = 0; i < 32; i++)
        expected_mac |= ((uint32_t)data[4 + i] << i); // CHANGED: offset by 4

    Bits stream(data.begin() + 36, data.end()); // CHANGED: offset by 36

    uint32_t actual_mac = simple_hash(stream);
    tampered = (actual_mac != expected_mac);

    Bits result;
    
    // Grab 1 bit, skip x padding bits
    for (size_t i = 0; i < stream.size();) {
        result.push_back(stream[i++]);
        i += x; 
    }
    return result;
}

// =====================================================================
//  Chunk-level encrypt / decrypt
// =====================================================================

static Bits encrypt_chunk(Bits s, Key &k, uint32_t pad_seed) {
    for (int i = 1; i <= 6; i++) {
        if (k.mask & (1u << i)) {
            Bits c(26);
            for (int j = 0; j < 26; j++)
                c[j] = (k.constants[i - 1] >> j) & 1;
            s = xor_bits(s, c);
        }
    }
    s = rotate_left(s, 3);
    s = swap_halves(s);
    s = reverse_bits(s);
    return apply_padding(s, pad_seed);
}

static Bits decrypt_chunk(Bits s, Key &k, bool &tampered) {
    s = remove_padding(s, tampered);

    s = reverse_bits(s);
    s = swap_halves(s);
    s = rotate_right(s, 3);

    for (int i = 6; i >= 1; i--) {
        if (k.mask & (1u << i)) {
            Bits c(26);
            for (int j = 0; j < 26; j++)
                c[j] = (k.constants[i - 1] >> j) & 1;
            s = xor_bits(s, c);
        }
    }
    return s;
}

// =====================================================================
//  PUBLIC API — called from Python
// =====================================================================

struct EncryptResult {
    std::string ciphertext;     // encrypted bytes
    int         num_bits_orig;  // original bit count (for trimming on decrypt)
    bool        ok;
};

struct DecryptResult {
    std::string plaintext;
    bool        tampered;       // true if any chunk failed MAC
};

/*
 * encrypt_data(data, seed)
 * data : raw bytes (the serialised tensor)
 * seed : symmetric key seed (shared secret)
 * Returns EncryptResult with the ciphertext bytes.
 */
EncryptResult encrypt_data(const std::string &data, uint32_t seed) {
    Key key = generate_key(seed);
    Bits bits = string_to_bits(data);
    int  orig_len = (int)bits.size();

    // Derive per-chunk padding seeds deterministically
    std::mt19937 pad_rng(seed ^ 0xDEADBEEF);

    Bits encrypted_all;

    for (size_t i = 0; i < bits.size(); i += 26) {
        Bits chunk;
        for (size_t j = i; j < i + 26 && j < bits.size(); j++)
            chunk.push_back(bits[j]);
        while (chunk.size() < 26) chunk.push_back(0);

        uint32_t ps = pad_rng();
        Bits enc = encrypt_chunk(chunk, key, ps);
        encrypted_all.insert(encrypted_all.end(), enc.begin(), enc.end());
    }

    EncryptResult r;
    r.ciphertext   = bits_to_string(encrypted_all);
    r.num_bits_orig = orig_len;
    r.ok            = true;
    return r;
}

/*
 * decrypt_data(ciphertext, seed, num_bits_orig, chunk_bit_len)
 * ciphertext     : encrypted bytes from encrypt_data
 * seed           : same seed used for encryption
 * num_bits_orig  : original plaintext bit count
 *
 * Because padding x is random-per-chunk, we need to know the exact
 * encrypted chunk sizes. We re-derive them from the deterministic seed.
 */
DecryptResult decrypt_data(const std::string &ciphertext, uint32_t seed,
                           int num_bits_orig) {
    Key key = generate_key(seed);
    Bits all_bits = string_to_bits(ciphertext);

    // Re-derive padding parameters to know chunk boundaries
    std::mt19937 pad_rng(seed ^ 0xDEADBEEF);

    int num_chunks = (num_bits_orig + 25) / 26;

    DecryptResult result;
    result.tampered = false;

    Bits final_bits;
    size_t offset = 0;

    for (int c = 0; c < num_chunks; c++) {
        // Re-derive x for this chunk (same sequence as encrypt)
        std::mt19937 chunk_rng(pad_rng());  // same seed as used in encrypt
        int x = chunk_rng() % 3 + 8; // CHANGED: Matches the 8, 9, 10 logic

        // Compute this chunk's encrypted bit length
        // 26 data bits. Each individual bit gets x padding bits.
        // padded_len = 26 * (1 + x)
        int padded_stream_len = 26 * (1 + x);
        int chunk_enc_bits = 36 + padded_stream_len; // CHANGED: 36 bit metadata

        if (offset + chunk_enc_bits > all_bits.size()) {
            // Truncated — take what we have
            chunk_enc_bits = (int)all_bits.size() - (int)offset;
        }

        Bits chunk(all_bits.begin() + offset, all_bits.begin() + offset + chunk_enc_bits);
        offset += chunk_enc_bits;

        bool chunk_tampered = false;
        Bits dec = decrypt_chunk(chunk, key, chunk_tampered);
        if (chunk_tampered)
            result.tampered = true;

        final_bits.insert(final_bits.end(), dec.begin(), dec.end());
    }

    // Trim to original length
    if ((int)final_bits.size() > num_bits_orig)
        final_bits.resize(num_bits_orig);

    result.plaintext = bits_to_string(final_bits);
    return result;
}

// =====================================================================
//  Pybind11 Bindings
// =====================================================================

#ifdef PYBIND11_BUILD

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(enc3, m) {
    m.doc() = "enc3 — Symmetric encryption library for Deep JSCC integration";

    py::class_<EncryptResult>(m, "EncryptResult")
        .def_property_readonly("ciphertext",
            [](const EncryptResult &r) {
                return py::bytes(r.ciphertext);
            })
        .def_readonly("num_bits_orig", &EncryptResult::num_bits_orig)
        .def_readonly("ok",            &EncryptResult::ok);

    py::class_<DecryptResult>(m, "DecryptResult")
        .def_property_readonly("plaintext",
            [](const DecryptResult &r) {
                return py::bytes(r.plaintext);
            })
        .def_readonly("tampered", &DecryptResult::tampered);

    m.def("encrypt_data",
        [](py::bytes data, uint32_t seed) {
            return encrypt_data(std::string(data), seed);
        },
        py::arg("data"), py::arg("seed"),
        "Encrypt a byte string with the given key seed.");

    m.def("decrypt_data",
        [](py::bytes ciphertext, uint32_t seed, int num_bits_orig) {
            return decrypt_data(std::string(ciphertext), seed, num_bits_orig);
        },
        py::arg("ciphertext"), py::arg("seed"), py::arg("num_bits_orig"),
        "Decrypt ciphertext. Returns DecryptResult with tampered flag.");
}

#endif  // PYBIND11_BUILD

// =====================================================================
//  Standalone test (compile with -DSTANDALONE_TEST)
// =====================================================================

#ifdef STANDALONE_TEST
#include <iostream>
#include <cassert>

int main() {
    std::string msg = "Hello, Deep JSCC!";
    uint32_t seed = 12345;

    auto enc = encrypt_data(msg, seed);
    std::cout << "Encrypted " << msg.size() << " bytes → "
              << enc.ciphertext.size() << " bytes\n";

    auto dec = decrypt_data(enc.ciphertext, seed, enc.num_bits_orig);
    std::cout << "Recovered: " << dec.plaintext << "\n";
    std::cout << "Tampered:  " << (dec.tampered ? "YES" : "no") << "\n";

    // Verify round-trip
    assert(dec.plaintext.substr(0, msg.size()) == msg);
    assert(!dec.tampered);
    std::cout << "✓ Round-trip OK\n";

    // Test tampering detection: flip a bit in ciphertext
    std::string corrupted = enc.ciphertext;
    corrupted[5] ^= 0x01;
    auto dec2 = decrypt_data(corrupted, seed, enc.num_bits_orig);
    std::cout << "After corruption — tampered: "
              << (dec2.tampered ? "YES ✓" : "no ✗") << "\n";

    return 0;
}
#endif  // STANDALONE_TEST