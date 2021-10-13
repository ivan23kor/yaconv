#pragma once

auto *cntx = bli_gks_query_cntx();
auto *data = new auxinfo_t;
auto bli_sgemm_ukr =
    (sgemm_ukr_ft)bli_cntx_get_l3_nat_ukr_dt(BLIS_FLOAT, BLIS_GEMM_UKR, cntx);

unsigned BLOCK_MR = bli_cntx_get_blksz(BLIS_MR, cntx)->v[0];
unsigned BLOCK_NR = bli_cntx_get_blksz(BLIS_NR, cntx)->v[0];
unsigned BLOCK_MC = bli_cntx_get_blksz(BLIS_MC, cntx)->v[0];
unsigned BLOCK_KC = bli_cntx_get_blksz(BLIS_KC, cntx)->v[0];
unsigned BLOCK_NC = bli_cntx_get_blksz(BLIS_NC, cntx)->v[0];

auto bli_packA_ukr =
    (spackm_cxk_ker_ft)bli_cntx_get_packm_ker_dt(BLIS_FLOAT, BLOCK_MR, cntx);
auto bli_packB_ukr =
    (spackm_cxk_ker_ft)bli_cntx_get_packm_ker_dt(BLIS_FLOAT, BLOCK_NR, cntx);
