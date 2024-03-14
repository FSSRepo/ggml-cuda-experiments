#ifndef __FLASH_LLAMA__
#define __FLASH_LLAMA__

// based on metal version
template<int D, int Q, int C> // D head size, Q queries per block, C cache items per block
static __global__ void flash_attn_ext_f16(
        const char* __restrict__ q,
        const char* __restrict__ k,
        const char* __restrict__ v,
        const char* __restrict__ mask,
        float* __restrict__ dst,
        float scale,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        int ne31,
        int nb31,
        int nb01,
        int nb02,
        int nb03,
        int nb11,
        int nb12,
        int nb13,
        int ne0,
        int ne1,
        int ne2,
        int ne3) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    const int num_warps = blockDim.y; // number of warps
    const int iq3 = blockIdx.z;
    const int iq2 = blockIdx.y;
    const int iq1 = blockIdx.x * Q;

    const int D16 = D/16;
    const int Q16 = Q/16;
    const int C16 = C/16;

    const int NW  = WARP_SIZE;
    const int SH  = (C + Q); // shared memory per simdgroup in (half)

    const int T  = D + num_warps*SH; // shared memory size per query in (half)
    const int T2 = T/2;              // shared memory size per query in (half2)
    const int C2 = C/2;
    const int D2 = D/2;

    extern __shared__  half __flash_attn_f16_shmem[];
    // pq
    half  * sq  = (half  *) (__flash_attn_f16_shmem +              0*D); // holds the query data
    half2 * sq2 = (half2 *) (__flash_attn_f16_shmem +              0*D); // same as above but in half2
    half  * ss  = (half  *) (__flash_attn_f16_shmem + warp_id*SH + 1*D); // scratch buffer for attention and diagonal matrix
    half2 * ss2 = (half2 *) (__flash_attn_f16_shmem + warp_id*SH + 1*D); // same as above but in half2

    AccumH zr;
    AccumH lo[Q16][D16];

    // load heads from Q to shared memory
#pragma unroll
    for (int j0 = 0; j0 < Q; j0 += num_warps) {
        const int j = j0 + warp_id;
        if (j >= Q) {
            break;
        }

        const float2 * q2 = (const float2 *) (q + ((iq1 + j)*nb01 + iq2*nb02 + iq3*nb03));

#pragma unroll
        for (int i0 = 0; i0 < D2; i0 += NW) {
            const int i = i0 + lane_id;
            if (i >= D2) {
                break;
            }

            if (iq1 + j < ne01) {
                sq2[j*T2 + i] = __float22half2_rn(q2[i]);
            } else {
                sq2[j*T2 + i] = make_half2(0.0, 0.0);
            }
        }
    }

    nvcuda::wmma::fill_fragment(zr, 0.0);

    // zero out lo
    for (int j = 0; j < Q16; ++j) {
        for (int i = 0; i < D16; ++i) {
            nvcuda::wmma::fill_fragment(lo[j][i], 0.0);
        }
    }

    // zero out shared memory SH
    for (int j = 0; j < Q; ++j) {
        for (int i0 = 0; i0 < SH; i0 += NW) {
            const int i = i0 + lane_id;
            if (i >= SH) {
                break;
            }

            ss[j*T + i] = 0.0;
        }
    }

    __syncthreads();

    {
        half S = __float2half(0.0f);
        half M[Q];

        for (int i = 0; i < Q; ++i) {
            M[i] = __float2half(-INFINITY);
        }

        // assume K and V are same shape
        const int ne22 = ne12;
        const int ne23 = ne13;

        const int nb21 = nb11;
        const int nb22 = nb12;
        const int nb23 = nb13;

        // broadcast
        const int rk2 = ne02/ne12;
        const int rk3 = ne03/ne13;

        const int rv2 = ne02/ne22;
        const int rv3 = ne03/ne23;

        // k indices
        const int ik2 = iq2 / rk2;
        const int ik3 = iq3 / rk3;

        // v indices
        const int iv2 = iq2 / rv2;
        const int iv3 = iq3 / rv3;

        // load the queries from shared memory into local memory
        MatrixA mq[Q16][D16];
        for (int j = 0; j < Q16; ++j) {
            for (int i = 0; i < D16; ++i) {
                nvcuda::wmma::load_matrix_sync(mq[j][i], sq + 16*j*T + i*16, T);
            }
        }

        // pointer to the mask
        const half * mp = mask ? (const half *) (mask + iq1*nb31) : nullptr;

        // prepare diagonal scale matrix
        MatrixB mscale;
        for (int i = 0; i < 16; ++i) {
            ss[i*T + i] = __float2half(scale);
        }
        nvcuda::wmma::load_matrix_sync(mscale, ss, T);

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < ne11; ic0 += C*num_warps) {
            const int ic = ic0 + warp_id*C;
            if (ic >= ne11) {
                break;
            }

            // Q*K^T
            {
#pragma unroll
                for (int cc = 0; cc < C16; ++cc) {
                    AccumH mqk[Q16];
                    for (int j = 0; j < Q16; ++j) {
                        nvcuda::wmma::fill_fragment(mqk[j], 0);
                    }

                    const half * pk = (const half *) ((const char *) k + ((ic + 16*cc)*nb11 + ik2*nb12 + ik3*nb13));

                    for (int i = 0; i < D16; ++i) {
                        MatrixBT mk; // transposed key
                        nvcuda::wmma::load_matrix_sync(mk, pk + i*16, nb11/sizeof(half));

                        for (int j = 0; j < Q16; ++j) {
                            nvcuda::wmma::mma_sync(mqk[j], mq[j][i], mk, mqk[j]);
                        }
                    }

                    // mqk = mqk*scale + mask
                    for (int j = 0; j < Q16; ++j) {
                        MatrixA mqka;
                        AccumH mm;

                        if (mp) {
                            nvcuda::wmma::load_matrix_sync(mm, mp + 16*j*(nb31/sizeof(half)) + ic + 16*cc, nb31/sizeof(half), nvcuda::wmma::mem_row_major);
                        }

                        // convert accumulator to matrix_a
                        nvcuda::wmma::store_matrix_sync(      ss + 16*j*T + 16*cc, mqk[j], T, nvcuda::wmma::mem_row_major);
                        nvcuda::wmma::load_matrix_sync (mqka, ss + 16*j*T + 16*cc, T);

                        nvcuda::wmma::mma_sync(mqk[j], mqka, mscale, mp ? mm : zr);
                        nvcuda::wmma::store_matrix_sync(ss + 16*j*T + 16*cc, mqk[j], T, nvcuda::wmma::mem_row_major);
                    }
                }
            }

            // used to detect blocks full of -INF
            half2 smax = make_half2(-INFINITY, -INFINITY);

            // online softmax
            for (int j = 0; j < Q; ++j) {
                const half m = M[j];

                for (int p0 = 0; p0 < C2; p0 += NW) {
                    const int p = p0 + lane_id;

                    const half2 s = ss2[j*T2 + p];

                    int seq_idx = p*2 + ic;

                    smax = __hmax2(smax, s);
                    M[j] = __hmax(M[j], __hmax(s.x, s.y));
                }

                M[j] = warp_reduce_max(M[j]);

                // if(j == 0 && lane_id == 0 && blockIdx.y == 0) {
                //     printf("max: %.4f, warp= %d\n", __half2float(M[0]), warp_id);
                // }

                // local sum
                half2 ls = make_half2(0.0f, 0.0f);
                half2 M2 = make_half2(M[j], M[j]);

                for (int p0 = 0; p0 < C2; p0 += NW) {
                    const int p = p0 + lane_id;

                    if(__hisinf(M[j]) == -1) {
                        ss2[j*T2 + p] = ls;
                        continue;
                    }

                    const half2 s = ss2[j*T2 + p];

                    const half2 vs = h2exp(s - M2);

                    ls += vs;

                    // the P matrix from the paper (Q rows, C columns)
                    ss2[j*T2 + p] = vs;
                }

                ls = warp_reduce_sum(ls);

                const half ms = hexp(m - M[j]);

                // create a QxQ diagonal matrix for rescaling the output
                if (lane_id == j && !__hisnan(ms)) {
                    ss[j*T + C + j] = ms;

                    S = S*ss[j*T + C + j] + ls.x + ls.y;
                }

                // if(j == 1 && lane_id == 0) {
                //     printf("sum: %.4f, warp= %d, j= %d\n", __half2float(S), warp_id, j);
                // }
            }

            smax = warp_reduce_max(smax);

            // if(lane_id == 0 && blockIdx.x) {
            //     printf("max 0: %.4f, max 1: %.4f, warp= %d\n", __half2float(smax.x), __half2float(smax.y), warp_id);
            // }

            // skip -INF blocks
            if (__hisinf(smax.x) == -1 && __hisinf(smax.y) == -1) {
                continue;
            }

            __syncthreads();

            // if(lane_id == 0 && warp_id == 1) {
            //     printf("diag MS: %d\n", warp_id);
            //     for(int j = 0; j < Q; j++) {
            //         for (int c = 0; c < 16; c ++) {
            //             printf("%.4f ", __half2float(ss[j*T + C + c]));
            //         }
            //         printf("\n");
            //     }
            // }
            // __syncthreads();

            // O = diag(ms)*O
            for (int j = 0; j < Q16; ++j) {
                MatrixA mm;
                MatrixB lob;

                nvcuda::wmma::load_matrix_sync(mm, ss + 16*j*T + C + 16*j, T);

                for (int i = 0; i < D16; ++i) {
                    // convert accumulator to matrix_b
                    nvcuda::wmma::store_matrix_sync(     ss + 16*j*T + C + 16*j, lo[j][i], T, nvcuda::wmma::mem_row_major);
                    nvcuda::wmma::load_matrix_sync (lob, ss + 16*j*T + C + 16*j, T);
                    nvcuda::wmma::mma_sync(lo[j][i], mm, lob, zr);

                    // if(warp_id == 1) {
                    //     nvcuda::wmma::store_matrix_sync(sq + 16*j*T + i*16, lo[j][i], T, nvcuda::wmma::mem_row_major);
                    // }
                }

                __syncthreads();
            }

            // restore zeros
            for (int j = 0; j < Q16; ++j) {
                nvcuda::wmma::store_matrix_sync(ss + 16*j*T + C + 16*j, zr, T, nvcuda::wmma::mem_row_major);
            }

            // O = O + (Q*K^T)*V
            {
                for (int cc = 0; cc < C16; ++cc) {
                    const half * pv = (const half *) ((const char *) v + ((ic + 16*cc)*nb21 + iv2*nb22 + iv3*nb23));

                    MatrixB mv[D16];
                    for (int i = 0; i < D16; ++i) {
                        nvcuda::wmma::load_matrix_sync(mv[i], pv + i*16, nb21/sizeof(half));
                    }

                    for (int j = 0; j < Q16; ++j) {
                        MatrixA ms;
                        nvcuda::wmma::load_matrix_sync(ms, ss + 16*j*T + 16*cc, T);
                        for (int i = 0; i < D16; ++i) {
                            nvcuda::wmma::mma_sync(lo[j][i], ms, mv[i], lo[j][i]);
                        }
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (lane_id < Q) {
            ss[lane_id*T + 0] = S;
            ss[lane_id*T + 1] = M[lane_id];

            // printf("S: %.4f, M: %.4f, warp= %d\n", __half2float(S), __half2float(M[lane_id]), warp_id);
        }
    }


    // reduce the warps sequentially
    for (int sg = 1; sg < num_warps; ++sg) {
        __syncthreads();

        // each simdgroup stores its output to shared memory, reusing sq
        if (warp_id == sg) {
            for (int j = 0; j < Q16; ++j) {
                for (int i = 0; i < D16; ++i) {
                    nvcuda::wmma::store_matrix_sync(sq + 16*j*T + i*16, lo[j][i], T, nvcuda::wmma::mem_row_major);
                }
            }
        }

        __syncthreads();

        // the first simdgroup accumulates the results from the other simdgroups
        if (warp_id == 0) {
            for (int j = lane_id; j < Q; j += NW) {
                const half S0 = ss[j*T +         0];
                const half S1 = ss[j*T + sg*SH + 0];

                const half M0 = ss[j*T +         1];
                const half M1 = ss[j*T + sg*SH + 1];

                const half M = __hmax(M0, M1);

                const half ms0 = hexp(M0 - M);
                const half ms1 = hexp(M1 - M);

                const half S = S0*ms0 + S1*ms1;

                ss[j*T + 0] = S;
                ss[j*T + 1] = M;

                ss[j*T + C + j        ] = ms0;
                ss[j*T + C + j + sg*SH] = ms1;
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            for (int j = 0; j < Q16; ++j) {
                MatrixA ms0;
                MatrixA ms1;
                MatrixB t;
                AccumH t2;

                nvcuda::wmma::load_matrix_sync(ms0, ss + 16*j*T + C + 16*j,         T);
                nvcuda::wmma::load_matrix_sync(ms1, ss + 16*j*T + C + 16*j + sg*SH, T);

                for (int i = 0; i < D16; ++i) {
                    nvcuda::wmma::load_matrix_sync(t, sq + 16*j*T + i*16, T);
                    nvcuda::wmma::mma_sync(t2, ms1, t, zr);

                    // convert accumulator to matrix_b
                    nvcuda::wmma::store_matrix_sync(   sq + 16*j*T + i*16, lo[j][i], T, nvcuda::wmma::mem_row_major);
                    nvcuda::wmma::load_matrix_sync (t, sq + 16*j*T + i*16, T);

                    nvcuda::wmma::mma_sync(lo[j][i], ms0, t, t2);
                }
            }
        }
    }

    if (warp_id == 0) {
        // store result to shared memory (reuse sq)
        for (int j = 0; j < Q16; ++j) {
            for (int i = 0; i < D16; ++i) {
                nvcuda::wmma::store_matrix_sync(sq + 16*j*T + i*16, lo[j][i], T, nvcuda::wmma::mem_row_major);
            }
        }

        // final rescale with 1/S and store to global memory
        for (int j = 0; j < Q && iq1 + j < ne01; ++j) {
            const half S = ss[j*T + 0];
            // if(blockIdx.y == 0 && lane_id == 0) {
            //     printf("Suma: %.4f\n", __half2float(S));
            // }

            for (int i0 = 0; i0 < D; i0 += NW) {
                const int i = i0 + lane_id;
                if (i >= D) {
                    break;
                }

                // printf("hdim %d = %.4f - %.4f\n", i, __half2float(sq[j*T + i]), __half2float(S));
                dst[(iq3*ne2*ne1 + iq2 + (iq1 + j)*ne1)*D + i] = __half2float(sq[j*T + i] / S);
            }
        }
    }
}

#endif