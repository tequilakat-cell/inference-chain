/*
 * inft_keccak.c — self-contained Keccak-256 implementation.
 *
 * Based on the "chasing-the-cycle" approach from Markku-Juhani Saarinen's
 * tiny_sha3 (MIT license), adapted for the Keccak-256 domain separator (0x01)
 * used by Ethereum rather than the SHA3-256 domain (0x06).
 *
 * Test vector:
 *   keccak256("") =
 *   c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
 */

#include "inft_keccak.h"
#include "pg_inft.h"

#include <string.h>

/* ── Constants ─────────────────────────────────────────────────────────────── */

static const uint64_t KECCAK_RC[24] = {
    UINT64_C(0x0000000000000001), UINT64_C(0x0000000000008082),
    UINT64_C(0x800000000000808A), UINT64_C(0x8000000080008000),
    UINT64_C(0x000000000000808B), UINT64_C(0x0000000080000001),
    UINT64_C(0x8000000080008081), UINT64_C(0x8000000000008009),
    UINT64_C(0x000000000000008A), UINT64_C(0x0000000000000088),
    UINT64_C(0x0000000080008009), UINT64_C(0x000000008000000A),
    UINT64_C(0x000000008000808B), UINT64_C(0x800000000000008B),
    UINT64_C(0x8000000000008089), UINT64_C(0x8000000000008003),
    UINT64_C(0x8000000000008002), UINT64_C(0x8000000000000080),
    UINT64_C(0x000000000000800A), UINT64_C(0x800000008000000A),
    UINT64_C(0x8000000080008081), UINT64_C(0x8000000000008080),
    UINT64_C(0x0000000080000001), UINT64_C(0x8000000080008008),
};

/* π lane permutation indices — the "spiral" traversal order */
static const int KECCAK_PILN[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

/* ρ rotation offsets, in the same spiral order as KECCAK_PILN */
static const int KECCAK_ROTC[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

/* ── Helper ────────────────────────────────────────────────────────────────── */

static inline uint64_t
rotl64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

/* ── Keccak-f[1600] permutation ───────────────────────────────────────────── */

static void
keccakf1600(uint64_t st[25])
{
    int      r, x, y, j;
    uint64_t t;
    uint64_t bc[5];

    for (r = 0; r < 24; r++)
    {
        /* θ */
        for (x = 0; x < 5; x++)
            bc[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (x = 0; x < 5; x++)
        {
            t = bc[(x+4) % 5] ^ rotl64(bc[(x+1) % 5], 1);
            for (y = 0; y < 25; y += 5)
                st[y + x] ^= t;
        }

        /* ρ and π (chasing the cycle) */
        t = st[1];
        for (j = 0; j < 24; j++)
        {
            x      = KECCAK_PILN[j];
            bc[0]  = st[x];
            st[x]  = rotl64(t, KECCAK_ROTC[j]);
            t      = bc[0];
        }

        /* χ */
        for (y = 0; y < 25; y += 5)
        {
            for (x = 0; x < 5; x++)
                bc[x] = st[y + x];
            for (x = 0; x < 5; x++)
                st[y + x] = bc[x] ^ ((~bc[(x+1) % 5]) & bc[(x+2) % 5]);
        }

        /* ι */
        st[0] ^= KECCAK_RC[r];
    }
}

/* ── Sponge construction ─────────────────────────────────────────────────── */

/*
 * keccak256 — absorb `len` bytes from `in`, squeeze 32 bytes into `out`.
 *
 * Rate = 136 bytes (1088 bits). Domain separator = 0x01 (Ethereum Keccak).
 * output = 32 bytes (256 bits).
 */
void
keccak256(const uint8_t *in, size_t len, uint8_t out[32])
{
    static const size_t RATE = 136;

    uint64_t st[25];
    uint8_t  temp[136];
    size_t   rsiz;
    size_t   i;

    memset(st, 0, sizeof(st));

    /* Absorb full blocks */
    rsiz = 0;
    while (len >= RATE - rsiz)
    {
        /* XOR `in` directly into state without copying */
        if (rsiz == 0 && len >= RATE)
        {
            /* Full block from `in` */
            for (i = 0; i < RATE / 8; i++)
            {
                uint64_t v = 0;
                const uint8_t *p = in + i * 8;
                v |= (uint64_t)p[0];
                v |= (uint64_t)p[1] << 8;
                v |= (uint64_t)p[2] << 16;
                v |= (uint64_t)p[3] << 24;
                v |= (uint64_t)p[4] << 32;
                v |= (uint64_t)p[5] << 40;
                v |= (uint64_t)p[6] << 48;
                v |= (uint64_t)p[7] << 56;
                st[i] ^= v;
            }
            in  += RATE;
            len -= RATE;
            keccakf1600(st);
        }
        else
            break;
    }

    /* Last block + pad10*1 padding */
    memset(temp, 0, RATE);
    if (len > 0)
        memcpy(temp, in, len);

    /* Keccak domain separator: 0x01 (not 0x06 which is SHA3) */
    temp[len]      ^= 0x01;
    temp[RATE - 1] ^= 0x80;

    for (i = 0; i < RATE / 8; i++)
    {
        uint64_t v = 0;
        const uint8_t *p = temp + i * 8;
        v |= (uint64_t)p[0];
        v |= (uint64_t)p[1] << 8;
        v |= (uint64_t)p[2] << 16;
        v |= (uint64_t)p[3] << 24;
        v |= (uint64_t)p[4] << 32;
        v |= (uint64_t)p[5] << 40;
        v |= (uint64_t)p[6] << 48;
        v |= (uint64_t)p[7] << 56;
        st[i] ^= v;
    }
    keccakf1600(st);

    /* Squeeze: write first 32 bytes of state (4 lanes) */
    for (i = 0; i < 4; i++)
    {
        out[i*8+0] = (uint8_t)(st[i]);
        out[i*8+1] = (uint8_t)(st[i] >> 8);
        out[i*8+2] = (uint8_t)(st[i] >> 16);
        out[i*8+3] = (uint8_t)(st[i] >> 24);
        out[i*8+4] = (uint8_t)(st[i] >> 32);
        out[i*8+5] = (uint8_t)(st[i] >> 40);
        out[i*8+6] = (uint8_t)(st[i] >> 48);
        out[i*8+7] = (uint8_t)(st[i] >> 56);
    }
}

/* ── SQL wrapper: inft_keccak256(bytea) → bytea ─────────────────────────── */

PG_FUNCTION_INFO_V1(inft_keccak256);
Datum
inft_keccak256(PG_FUNCTION_ARGS)
{
    bytea         *input  = PG_GETARG_BYTEA_PP(0);
    const uint8_t *data   = (const uint8_t *) VARDATA_ANY(input);
    size_t         dlen   = VARSIZE_ANY_EXHDR(input);
    uint8_t        hash[32];
    bytea         *result;

    keccak256(data, dlen, hash);

    result = (bytea *) palloc(VARHDRSZ + 32);
    SET_VARSIZE(result, VARHDRSZ + 32);
    memcpy(VARDATA(result), hash, 32);

    PG_RETURN_BYTEA_P(result);
}
