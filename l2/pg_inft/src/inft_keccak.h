/*
 * inft_keccak.h — public C API for the embedded Keccak-256 implementation.
 *
 * This is Keccak-256 (domain separation byte 0x01), NOT SHA3-256 (0x06).
 *
 * Test vector:
 *   keccak256(b"") =
 *   c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
 *
 * The permutation is Keccak-f[1600] with 24 rounds operating on a 5×5 matrix
 * of 64-bit lanes, rate = 1088 bits (136 bytes), capacity = 512 bits.
 */

#ifndef INFT_KECCAK_H
#define INFT_KECCAK_H

#include <stdint.h>
#include <stddef.h>

/*
 * keccak256 — hash `len` bytes from `in` and write 32 bytes to `out`.
 *
 * `out` must point to at least 32 bytes of writable storage.
 * `in`  may be NULL only when `len` == 0.
 */
void keccak256(const uint8_t *in, size_t len, uint8_t out[32]);

#endif /* INFT_KECCAK_H */
