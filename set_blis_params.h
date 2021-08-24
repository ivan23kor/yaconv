#pragma once

auto *cntx = bli_gks_query_cntx();
auto *data = new auxinfo_t;
auto blisGemmUKR = (sgemm_ukr_ft)bli_cntx_get_l3_nat_ukr_dt(BLIS_FLOAT, BLIS_GEMM_UKR, cntx);

unsigned BLOCK_MR = bli_cntx_get_blksz(BLIS_MR, cntx)->v[0];
unsigned BLOCK_NR = bli_cntx_get_blksz(BLIS_NR, cntx)->v[0];
unsigned BLOCK_MC = bli_cntx_get_blksz(BLIS_MC, cntx)->v[0];
unsigned BLOCK_KC = bli_cntx_get_blksz(BLIS_KC, cntx)->v[0];
unsigned BLOCK_NC = bli_cntx_get_blksz(BLIS_NC, cntx)->v[0];
