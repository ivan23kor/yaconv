#pragma once

// Internal BLIS structures, used in sgemm_ukr and to retrieve block sizes
auto *cntx = bli_gks_query_cntx();
auto *data = new auxinfo_t;

// BLIS block sizes
int BLOCK_MR = bli_cntx_get_blksz(BLIS_MR, cntx)->v[BLIS_FLOAT];
int BLOCK_NR = bli_cntx_get_blksz(BLIS_NR, cntx)->v[BLIS_FLOAT];
int BLOCK_MC = bli_cntx_get_blksz(BLIS_MC, cntx)->v[BLIS_FLOAT];
int BLOCK_KC = bli_cntx_get_blksz(BLIS_KC, cntx)->v[BLIS_FLOAT];
int BLOCK_NC = bli_cntx_get_blksz(BLIS_NC, cntx)->v[BLIS_FLOAT];

// sgemm microkernel and packing functions from BLIS
auto bli_sgemm_ukr =
    (sgemm_ukr_ft)bli_cntx_get_l3_nat_ukr_dt(BLIS_FLOAT, BLIS_GEMM_UKR, cntx);
auto bli_packA_ukr = (spackm_cxk_ker_ft)bli_cntx_get_packm_ker_dt(
    BLIS_FLOAT, (l1mkr_t)BLOCK_MR, cntx);
auto bli_packB_ukr = (spackm_cxk_ker_ft)bli_cntx_get_packm_ker_dt(
    BLIS_FLOAT, (l1mkr_t)BLOCK_NR, cntx);

// Wrappers around BLIS functions, these hide some internal BLIS parameters
inline void sgemm_ukr(int KC, float *Alpha, float *Ar, float *Br, float *Beta,
                      float *Cr, int rsc, int csc) {
  bli_sgemm_ukr(KC, Alpha, Ar, Br, Beta, Cr, rsc, csc, data, cntx);
}

inline void packAPanel(float *APanel, float *PackPanel, int MR,
                       int KC, int rsa, int csa, int incp) {
  bli_packA_ukr(BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS, MR, KC, KC, bli_s1,
                APanel, rsa, csa, PackPanel, incp, cntx);
}

inline void packBPanel(float *BPanel, float *PackPanel, int NR,
                       int KC, int rsb, int csb, int incp) {
  bli_packB_ukr(BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS, NR, KC, KC, bli_s1,
                BPanel, rsb, csb, PackPanel, incp, cntx);
}
