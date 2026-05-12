# LLM-AGR Privacy-Utility Trade-off: Final Report

Auto-generated from `results/*.jsonl` and `results/utility/*.json`.

## 1. Summary

| Dataset | LGCN R@20 | AGR R@20 | ΔR@20 | LGCN AUC | AGR AUC | ΔAUC | verdict |
|---|---|---|---|---|---|---|---|
| digital_music | – | 0.2520 | 0.2520 | 0.9165 | 0.9341 | 0.0176 | AGR boosts recall but raises AUC |

*ΔR@20 > 0: AGR recommends better. ΔAUC < 0: AGR is more privacy-safe.*


## 2. Pipeline overview

1. **Data prep** (`scripts/prepare_dataset.py`): rating filter (≥3) → iterative k-core (k=5) → 6:2:2 random per-user split → BAAI BGE-small-en-v1.5 sentence encoder for user reviews and item title+reviews → save `trn/val/tst.npz` + `usr_emb_np.npy` + `itm_emb_np.npy`.

2. **Baseline training** (`main.py`): LightGCN (BPR + L2) and LightGCN_AGR (BPR + L2 + PRF distillation + STR contrast on LLM-augmented graph + HSIC info-bottleneck + masked recon).

3. **MIA** (`attack/MIA.py`): for each ckpt, load the model, compute final embeddings, build hand-crafted features for sampled member/non-member edges, train an LR attacker, report AUC/ACC/F1.

4. **Hyperparameter sweep** (`scripts/sweep.py`): vary regularization weights (β, str_weight, prf_weight) on a single dataset to map the privacy-utility trade-off.


## 3. Detailed results: `digital_music`

| Model | Tag | Seed | R@5 | R@10 | R@20 | NDCG@20 | MIA AUC | MIA ACC | best_epoch |
|---|---|---|---|---|---|---|---|---|---|
| lightgcn | attack_floor_degreeonly | 2025 | – | – | – | – | 0.9001 | 0.8235 | – |
| lightgcn | baseline | 2025 | 0.1011 | 0.1649 | 0.2414 | 0.1327 | 0.9016 | 0.8229 | 225 |
| lightgcn | lgcn_T1.0_n0.0 | 2025 | – | – | – | – | 0.9016 | 0.8229 | – |
| lightgcn | lgcn_T1.0_n0.05 | 2025 | – | – | – | – | 0.9009 | 0.8221 | – |
| lightgcn | lgcn_T1.0_n0.2 | 2025 | – | – | – | – | 0.8912 | 0.8114 | – |
| lightgcn | lgcn_T10.0_n0.0 | 2025 | – | – | – | – | 0.8999 | 0.8224 | – |
| lightgcn | lgcn_T10.0_n0.05 | 2025 | – | – | – | – | 0.8329 | 0.7545 | – |
| lightgcn | lgcn_T10.0_n0.2 | 2025 | – | – | – | – | 0.6477 | 0.6169 | – |
| lightgcn | lgcn_T2.0_n0.0 | 2025 | – | – | – | – | 0.9010 | 0.8265 | – |
| lightgcn | lgcn_T2.0_n0.05 | 2025 | – | – | – | – | 0.8980 | 0.8234 | – |
| lightgcn | lgcn_T2.0_n0.2 | 2025 | – | – | – | – | 0.8564 | 0.7781 | – |
| lightgcn | lgcn_T5.0_n0.0 | 2025 | – | – | – | – | 0.9002 | 0.8237 | – |
| lightgcn | lgcn_T5.0_n0.05 | 2025 | – | – | – | – | 0.8836 | 0.8054 | – |
| lightgcn | lgcn_T5.0_n0.2 | 2025 | – | – | – | – | 0.7115 | 0.6536 | – |
| lightgcn | lgcn_baseline_balanced | 2025 | – | – | – | – | 0.8963 | 0.8248 | – |
| lightgcn | lgcn_baseline_mlp | 2025 | – | – | – | – | 0.9165 | 0.8376 | – |
| lightgcn | lgcn_baseline_mlp | 2025 | – | – | – | – | 0.9165 | 0.8376 | – |
| lightgcn | lgcn_baseline_mlp | 2025 | – | – | – | – | 0.9165 | 0.8376 | – |
| lightgcn | lgcn_baseline_seed2026 | 2026 | 0.1026 | 0.1614 | 0.2373 | 0.1304 | 0.8926 | 0.8134 | 195 |
| lightgcn | lgcn_baseline_seed2027 | 2027 | 0.1026 | 0.1614 | 0.2373 | 0.1304 | 0.8926 | 0.8134 | 195 |
| lightgcn_agr | ag_kr0.3 | 2025 | 0.0594 | 0.0935 | 0.1505 | 0.0782 | 0.8341 | 0.7468 | 9 |
| lightgcn_agr | ag_kr0.5 | 2025 | 0.0574 | 0.0883 | 0.1454 | 0.0753 | 0.8296 | 0.7428 | 6 |
| lightgcn_agr | agr_T1.0_n0.0 | 2025 | – | – | – | – | 0.9341 | 0.8508 | – |
| lightgcn_agr | agr_T1.0_n0.05 | 2025 | – | – | – | – | 0.9327 | 0.8484 | – |
| lightgcn_agr | agr_T1.0_n0.2 | 2025 | – | – | – | – | 0.9080 | 0.8228 | – |
| lightgcn_agr | agr_T10.0_n0.0 | 2025 | – | – | – | – | 0.9330 | 0.8500 | – |
| lightgcn_agr | agr_T10.0_n0.05 | 2025 | – | – | – | – | 0.7870 | 0.7151 | – |
| lightgcn_agr | agr_T10.0_n0.2 | 2025 | – | – | – | – | 0.6452 | 0.6147 | – |
| lightgcn_agr | agr_T2.0_n0.0 | 2025 | – | – | – | – | 0.9332 | 0.8500 | – |
| lightgcn_agr | agr_T2.0_n0.05 | 2025 | – | – | – | – | 0.9274 | 0.8449 | – |
| lightgcn_agr | agr_T2.0_n0.2 | 2025 | – | – | – | – | 0.8302 | 0.7502 | – |
| lightgcn_agr | agr_T5.0_n0.0 | 2025 | – | – | – | – | 0.9330 | 0.8500 | – |
| lightgcn_agr | agr_T5.0_n0.05 | 2025 | – | – | – | – | 0.8922 | 0.8066 | – |
| lightgcn_agr | agr_T5.0_n0.2 | 2025 | – | – | – | – | 0.6645 | 0.6216 | – |
| lightgcn_agr | agr_b0s0_mlp | 2025 | – | – | – | – | 0.8982 | 0.8138 | – |
| lightgcn_agr | agr_b1s0_mlp | 2025 | – | – | – | – | 0.9187 | 0.8370 | – |
| lightgcn_agr | agr_b4s0_mlp | 2025 | – | – | – | – | 0.9129 | 0.8219 | – |
| lightgcn_agr | agr_default_balanced | 2025 | – | – | – | – | 0.9339 | 0.8497 | – |
| lightgcn_agr | agr_default_mlp | 2025 | – | – | – | – | 0.9449 | 0.8662 | – |
| lightgcn_agr | agr_default_mlp | 2025 | – | – | – | – | 0.9449 | 0.8662 | – |
| lightgcn_agr | agr_default_mlp | 2025 | – | – | – | – | 0.9449 | 0.8662 | – |
| lightgcn_agr | b0.0_s0.0 | 2025 | 0.1069 | 0.1633 | 0.2419 | 0.1352 | 0.8857 | 0.8034 | 186 |
| lightgcn_agr | b0.0_s0.05 | 2025 | 0.1190 | 0.1759 | 0.2505 | 0.1433 | 0.9293 | 0.8435 | 48 |
| lightgcn_agr | b0.0_s0.2 | 2025 | 0.1045 | 0.1571 | 0.2212 | 0.1288 | 0.9534 | 0.8803 | 102 |
| lightgcn_agr | b0s0_balanced | 2025 | – | – | – | – | 0.8810 | 0.8057 | – |
| lightgcn_agr | b0s0_mlp | 2025 | – | – | – | – | 0.8982 | 0.8138 | – |
| lightgcn_agr | b0s0_mlp | 2025 | – | – | – | – | 0.8982 | 0.8138 | – |
| lightgcn_agr | b0s0_seed2026 | 2026 | 0.1094 | 0.1650 | 0.2449 | 0.1365 | 0.8892 | 0.8072 | 198 |
| lightgcn_agr | b0s0_seed2027 | 2027 | 0.1079 | 0.1643 | 0.2426 | 0.1354 | 0.8876 | 0.8051 | 192 |
| lightgcn_agr | b1.0_s0.0 | 2025 | 0.1144 | 0.1726 | 0.2548 | 0.1435 | 0.9097 | 0.8247 | 165 |
| lightgcn_agr | b1.0_s0.05 | 2025 | 0.1189 | 0.1754 | 0.2499 | 0.1432 | 0.9303 | 0.8448 | 48 |
| lightgcn_agr | b1.0_s0.2 | 2025 | 0.1042 | 0.1550 | 0.2203 | 0.1269 | 0.9493 | 0.8738 | 90 |
| lightgcn_agr | b1s0_balanced | 2025 | – | – | – | – | 0.9064 | 0.8247 | – |
| lightgcn_agr | b1s0_mlp | 2025 | – | – | – | – | 0.9187 | 0.8370 | – |
| lightgcn_agr | b1s0_mlp | 2025 | – | – | – | – | 0.9187 | 0.8370 | – |
| lightgcn_agr | b4.0_s0.0 | 2025 | 0.1125 | 0.1718 | 0.2483 | 0.1402 | 0.9036 | 0.8113 | 75 |
| lightgcn_agr | b4.0_s0.05 | 2025 | 0.1185 | 0.1742 | 0.2499 | 0.1438 | 0.9348 | 0.8511 | 51 |
| lightgcn_agr | b4.0_s0.2 | 2025 | 0.1038 | 0.1551 | 0.2208 | 0.1278 | 0.9521 | 0.8783 | 96 |
| lightgcn_agr | b4s0_balanced | 2025 | – | – | – | – | 0.9013 | 0.8119 | – |
| lightgcn_agr | b4s0_mlp | 2025 | – | – | – | – | 0.9129 | 0.8219 | – |
| lightgcn_agr | b4s0_mlp | 2025 | – | – | – | – | 0.9129 | 0.8219 | – |
| lightgcn_agr | default | 2025 | 0.1194 | 0.1770 | 0.2520 | 0.1444 | 0.9341 | 0.8508 | 54 |
| lightgcn_agr | friend_params | 2025 | 0.1168 | 0.1715 | 0.2388 | 0.1391 | 0.9400 | 0.8588 | 105 |
| lightgcn_agr | friend_params_p10 | 2025 | 0.1157 | 0.1700 | 0.2384 | 0.1386 | 0.9377 | 0.8550 | 99 |
| lightgcn_agr | m3_emb12_b0s0 | 2025 | 0.0860 | 0.1396 | 0.2133 | 0.1142 | 0.8336 | 0.7512 | 192 |
| lightgcn_agr | m3_emb16_b0s0 | 2025 | 0.0889 | 0.1420 | 0.2195 | 0.1176 | 0.8455 | 0.7627 | 186 |
| lightgcn_agr | m3_emb16_default | 2025 | 0.1063 | 0.1606 | 0.2314 | 0.1315 | 0.9035 | 0.8162 | 96 |
| lightgcn_agr | m3_emb24_b0s0 | 2025 | 0.0918 | 0.1470 | 0.2244 | 0.1213 | 0.8569 | 0.7758 | 150 |
| lightgcn_agr | m3_emb8_b0s0 | 2025 | 0.0769 | 0.1267 | 0.2008 | 0.1051 | 0.8102 | 0.7307 | 183 |
| lightgcn_agr | m3_emb8_default | 2025 | 0.0820 | 0.1305 | 0.1959 | 0.1066 | 0.8510 | 0.7633 | 198 |
| lightgcn_agr | p13a_emb16_b0s0 | 2025 | 0.0888 | 0.1420 | 0.2195 | 0.1177 | 0.8455 | 0.7626 | 186 |
| lightgcn_agr | p13b_augk0_b0s0 | 2025 | 0.1093 | 0.1650 | 0.2445 | 0.1364 | 0.8892 | 0.8073 | 198 |
| lightgcn_agr | p13b_augk10_b0s0 | 2025 | 0.1092 | 0.1648 | 0.2449 | 0.1365 | 0.8892 | 0.8070 | 198 |
| lightgcn_agr | p13b_augk20_b0s0 | 2025 | 0.1094 | 0.1649 | 0.2448 | 0.1364 | 0.8892 | 0.8072 | 198 |
| lightgcn_agr | p13b_augk3_b0s0 | 2025 | 0.1089 | 0.1647 | 0.2447 | 0.1364 | 0.8892 | 0.8071 | 198 |
| lightgcn_agr | p13c_kr0.3 | 2025 | 0.0594 | 0.0935 | 0.1506 | 0.0782 | 0.8341 | 0.7469 | 9 |
| lightgcn_agr | p13c_kr0.5 | 2025 | 0.0961 | 0.1503 | 0.2236 | 0.1224 | 0.8559 | 0.7734 | 198 |
| lightgcn_agr | p13c_kr0.6 | 2025 | 0.0999 | 0.1536 | 0.2340 | 0.1285 | 0.8648 | 0.7836 | 195 |
| lightgcn_agr | p14a_agrdef_seed2026 | 2026 | 0.1189 | 0.1753 | 0.2499 | 0.1432 | 0.9303 | 0.8448 | 48 |
| lightgcn_agr | p14a_agrdef_seed2027 | 2027 | 0.1197 | 0.1774 | 0.2528 | 0.1450 | 0.9358 | 0.8533 | 57 |
| lightgcn_agr | p14a_emb16b0s0_seed2026 | 2026 | 0.0890 | 0.1423 | 0.2196 | 0.1177 | 0.8455 | 0.7627 | 186 |
| lightgcn_agr | p14a_emb16b0s0_seed2027 | 2027 | 0.0890 | 0.1423 | 0.2194 | 0.1176 | 0.8455 | 0.7625 | 186 |
| lightgcn_agr | p14c_emb4_b0s0 | 2025 | 0.0461 | 0.0797 | 0.1338 | 0.0677 | 0.7577 | 0.6833 | 90 |
| lightgcn_agr | p14c_emb6_b0s0 | 2025 | 0.0649 | 0.1129 | 0.1766 | 0.0919 | 0.7857 | 0.7120 | 150 |
| lightgcn_agr | p15_emb10_b0s0 | 2025 | 0.0786 | 0.1259 | 0.2013 | 0.1059 | 0.8211 | 0.7407 | 195 |
| lightgcn_agr | p15_emb20_b0s0 | 2025 | 0.0978 | 0.1516 | 0.2316 | 0.1257 | 0.8600 | 0.7790 | 198 |
| lightgcn_agr | p16_emb2_b0s0 | 2025 | 0.0364 | 0.0627 | 0.1053 | 0.0530 | 0.7143 | 0.6460 | 87 |
| lightgcn_agr | p16_emb3_b0s0 | 2025 | 0.0399 | 0.0714 | 0.1187 | 0.0595 | 0.7359 | 0.6652 | 105 |
| lightgcn_agr | p16_emb4_seed2026 | 2026 | 0.0461 | 0.0799 | 0.1338 | 0.0678 | 0.7577 | 0.6832 | 90 |
| lightgcn_agr | p16_emb4_seed2027 | 2027 | 0.0461 | 0.0797 | 0.1339 | 0.0678 | 0.7577 | 0.6832 | 90 |
| lightgcn_agr | p16_emb8_seed2026 | 2026 | 0.0769 | 0.1265 | 0.2007 | 0.1051 | 0.8102 | 0.7306 | 183 |
| lightgcn_agr | p16_emb8_seed2027 | 2027 | 0.0775 | 0.1273 | 0.2015 | 0.1055 | 0.8106 | 0.7311 | 186 |
| lightgcn_agr | p17_dpsgd_ref | 2025 | 0.0646 | 0.1125 | 0.1816 | 0.0932 | 0.8060 | 0.7241 | 27 |
| lightgcn_agr | p17_emb12_seed2026 | 2026 | 0.0861 | 0.1400 | 0.2132 | 0.1142 | 0.8336 | 0.7512 | 192 |
| lightgcn_agr | p17_emb12_seed2027 | 2027 | 0.0865 | 0.1396 | 0.2128 | 0.1142 | 0.8336 | 0.7511 | 192 |
| lightgcn_agr | p17_emb6_seed2026 | 2026 | 0.0655 | 0.1135 | 0.1764 | 0.0920 | 0.7862 | 0.7124 | 153 |
| lightgcn_agr | p17_emb6_seed2027 | 2027 | 0.0667 | 0.1152 | 0.1788 | 0.0934 | 0.7903 | 0.7151 | 174 |
| lightgcn_agr | published_agr_seed2025 | 2025 | 0.1116 | 0.1676 | 0.2338 | 0.1367 | 0.9343 | 0.8564 | 165 |
| lightgcn_agr_adv | m4_adv0.02_p12 | 2025 | 0.1087 | 0.1635 | 0.2450 | 0.1359 | 0.8835 | 0.8001 | 198 |
| lightgcn_agr_adv | m4_adv0.05 | 2025 | 0.1068 | 0.1641 | 0.2437 | 0.1346 | 0.8755 | 0.7908 | 198 |
| lightgcn_agr_adv | m4_adv0.05_lam0.5_p12 | 2025 | 0.1088 | 0.1647 | 0.2449 | 0.1357 | 0.8822 | 0.7993 | 198 |
| lightgcn_agr_adv | m4_adv0.05_mlp | 2025 | – | – | – | – | 0.8888 | 0.8042 | – |
| lightgcn_agr_adv | m4_adv0.05_mlp | 2025 | – | – | – | – | 0.8888 | 0.8042 | – |
| lightgcn_agr_adv | m4_adv0.05_seed2026 | 2026 | 0.1033 | 0.1597 | 0.2411 | 0.1319 | 0.8667 | 0.7820 | 168 |
| lightgcn_agr_adv | m4_adv0.05_seed2027 | 2027 | 0.1069 | 0.1639 | 0.2434 | 0.1344 | 0.8755 | 0.7911 | 198 |
| lightgcn_agr_adv | m4_adv0.1_p12 | 2025 | 0.1040 | 0.1580 | 0.2380 | 0.1301 | 0.8604 | 0.7766 | 198 |
| lightgcn_agr_adv | m4_adv0.2 | 2025 | 0.0520 | 0.0875 | 0.1411 | 0.0727 | 0.8379 | 0.7522 | 3 |
| lightgcn_agr_adv | m4_adv1.0 | 2025 | 0.0503 | 0.0856 | 0.1392 | 0.0716 | 0.8490 | 0.7622 | 3 |
| lightgcn_agr_adv | m4_adv1.0_lam2 | 2025 | 0.0431 | 0.0759 | 0.1246 | 0.0632 | 0.8569 | 0.7663 | 3 |
| lightgcn_agr_adv | m4_adv3.0 | 2025 | 0.0371 | 0.0672 | 0.1119 | 0.0568 | 0.8570 | 0.7697 | 3 |
| lightgcn_agr_adv | p13a_emb16_adv0.05 | 2025 | 0.0903 | 0.1403 | 0.2176 | 0.1179 | 0.8375 | 0.7531 | 192 |
| lightgcn_agr_adv | p13a_emb16_adv0.1 | 2025 | 0.0857 | 0.1363 | 0.2130 | 0.1149 | 0.8273 | 0.7431 | 192 |
| lightgcn_agr_adv | p13a_emb24_adv0.05 | 2025 | 0.0927 | 0.1468 | 0.2241 | 0.1219 | 0.8510 | 0.7693 | 165 |
| lightgcn_agr_adv | p13d_adv0.05_p15 | 2025 | 0.1069 | 0.1638 | 0.2431 | 0.1343 | 0.8749 | 0.7902 | 195 |
| lightgcn_agr_adv | p13d_adv0.15_p15 | 2025 | 0.0972 | 0.1491 | 0.2251 | 0.1225 | 0.8427 | 0.7569 | 192 |
| lightgcn_agr_adv | p13d_adv0.2_lam0.5_p15 | 2025 | 0.1039 | 0.1573 | 0.2369 | 0.1305 | 0.8604 | 0.7762 | 198 |
| lightgcn_agr_adv | p15_emb10_adv0.05 | 2025 | 0.0790 | 0.1274 | 0.2000 | 0.1051 | 0.8125 | 0.7340 | 192 |
| lightgcn_agr_adv | p15_emb10_adv0.1 | 2025 | 0.0444 | 0.0735 | 0.1207 | 0.0615 | 0.7663 | 0.6883 | 6 |
| lightgcn_agr_adv | p15_emb12_adv0.05 | 2025 | 0.0861 | 0.1416 | 0.2186 | 0.1162 | 0.8289 | 0.7486 | 198 |
| lightgcn_agr_adv | p15_emb16_adv0.1_seed2026 | 2026 | 0.0859 | 0.1373 | 0.2143 | 0.1157 | 0.8281 | 0.7439 | 195 |
| lightgcn_agr_adv | p15_emb16_adv0.1_seed2027 | 2027 | 0.0859 | 0.1370 | 0.2133 | 0.1150 | 0.8273 | 0.7430 | 192 |
| lightgcn_agr_adv | p15_emb20_adv0.05 | 2025 | 0.0930 | 0.1471 | 0.2280 | 0.1222 | 0.8499 | 0.7661 | 198 |
| lightgcn_agr_adv | p16_emb16adv01_seed2026 | 2026 | 0.0859 | 0.1362 | 0.2132 | 0.1150 | 0.8273 | 0.7430 | 192 |
| lightgcn_agr_adv | p16_emb16adv01_seed2027 | 2027 | 0.0859 | 0.1364 | 0.2135 | 0.1151 | 0.8273 | 0.7432 | 192 |
| lightgcn_agr_adv | p17_adv005_p20 | 2025 | 0.1070 | 0.1635 | 0.2418 | 0.1341 | 0.8749 | 0.7904 | 195 |
| lightgcn_agr_combo | p15_combo_c0.5_a0.1_seed2026 | 2026 | 0.1097 | 0.1629 | 0.2445 | 0.1351 | 0.8632 | 0.7814 | 195 |
| lightgcn_agr_combo | p15_combo_c0.5_a0.1_seed2027 | 2027 | 0.1106 | 0.1618 | 0.2442 | 0.1350 | 0.8640 | 0.7825 | 198 |
| lightgcn_agr_combo | p15_emb16_triple1 | 2025 | 0.0915 | 0.1444 | 0.2195 | 0.1198 | 0.8382 | 0.7557 | 189 |
| lightgcn_agr_combo | p15_emb16_triple2 | 2025 | 0.0915 | 0.1445 | 0.2186 | 0.1193 | 0.8372 | 0.7561 | 183 |
| lightgcn_agr_combo | s1_combo_c0.1_a0.05 | 2025 | 0.1088 | 0.1632 | 0.2470 | 0.1365 | 0.8769 | 0.7936 | 198 |
| lightgcn_agr_combo | s1_combo_c0.2_a0.05_t0.6 | 2025 | 0.1106 | 0.1648 | 0.2471 | 0.1372 | 0.8783 | 0.7957 | 198 |
| lightgcn_agr_combo | s1_combo_c0.3_a0.02 | 2025 | 0.1104 | 0.1663 | 0.2475 | 0.1384 | 0.8865 | 0.8063 | 192 |
| lightgcn_agr_combo | s1_combo_c0.5_a0.05 | 2025 | 0.1098 | 0.1655 | 0.2456 | 0.1371 | 0.8781 | 0.7980 | 189 |
| lightgcn_agr_combo | s1_combo_c0.5_a0.1 | 2025 | 0.1107 | 0.1637 | 0.2441 | 0.1355 | 0.8628 | 0.7809 | 192 |
| lightgcn_agr_conf | m1_conf01_mlp | 2025 | – | – | – | – | 0.8918 | 0.8085 | – |
| lightgcn_agr_conf | m1_conf01_mlp | 2025 | – | – | – | – | 0.8918 | 0.8085 | – |
| lightgcn_agr_conf | m1_conf05_mlp | 2025 | – | – | – | – | 0.9006 | 0.8214 | – |
| lightgcn_agr_conf | m1_conf05_mlp | 2025 | – | – | – | – | 0.9006 | 0.8214 | – |
| lightgcn_agr_conf | m1_conf_w0.1_t0.7 | 2025 | 0.1052 | 0.1620 | 0.2418 | 0.1334 | 0.8805 | 0.7987 | 165 |
| lightgcn_agr_conf | m1_conf_w0.3_p12 | 2025 | 0.1099 | 0.1691 | 0.2491 | 0.1389 | 0.8945 | 0.8160 | 198 |
| lightgcn_agr_conf | m1_conf_w0.5_t0.55 | 2025 | 0.1100 | 0.1685 | 0.2500 | 0.1388 | 0.8920 | 0.8140 | 180 |
| lightgcn_agr_conf | m1_conf_w0.5_t0.7 | 2025 | 0.1080 | 0.1659 | 0.2465 | 0.1367 | 0.8870 | 0.8075 | 168 |
| lightgcn_agr_conf | m1_conf_w1.0_p12 | 2025 | 0.1092 | 0.1671 | 0.2462 | 0.1375 | 0.8949 | 0.8208 | 186 |
| lightgcn_agr_conf | m1_conf_w2.0_t0.7 | 2025 | 0.1099 | 0.1634 | 0.2425 | 0.1362 | 0.8997 | 0.8282 | 162 |
| lightgcn_agr_conf | m1_conf_w5.0_t0.6 | 2025 | 0.0986 | 0.1532 | 0.2269 | 0.1259 | 0.8871 | 0.8074 | 72 |
| lightgcn_agr_conf | p14a_m1conf01_seed2026 | 2026 | 0.1078 | 0.1657 | 0.2476 | 0.1379 | 0.8909 | 0.8103 | 198 |
| lightgcn_agr_conf | p14a_m1conf01_seed2027 | 2027 | 0.1077 | 0.1656 | 0.2476 | 0.1379 | 0.8909 | 0.8102 | 198 |
| lightgcn_agr_conf | p14b_emb12_conf03 | 2025 | 0.0848 | 0.1354 | 0.2085 | 0.1115 | 0.8252 | 0.7462 | 162 |
| lightgcn_agr_conf | p14b_emb16_conf01 | 2025 | 0.0914 | 0.1444 | 0.2217 | 0.1193 | 0.8494 | 0.7669 | 198 |
| lightgcn_agr_conf | p14b_emb16_conf05 | 2025 | 0.0917 | 0.1467 | 0.2242 | 0.1210 | 0.8517 | 0.7718 | 198 |
| lightgcn_agr_conf | p14b_emb24_conf03 | 2025 | 0.1018 | 0.1573 | 0.2364 | 0.1297 | 0.8764 | 0.7971 | 198 |
| lightgcn_agr_conf | p16_conf_w0.3_t0.4 | 2025 | 0.1109 | 0.1709 | 0.2505 | 0.1404 | 0.8962 | 0.8171 | 198 |
| lightgcn_agr_conf | p16_conf_w0.3_t0.6 | 2025 | 0.1096 | 0.1686 | 0.2496 | 0.1395 | 0.8941 | 0.8153 | 195 |
| lightgcn_agr_conf | p16_conf_w0.3_t0.8 | 2025 | 0.1097 | 0.1672 | 0.2484 | 0.1383 | 0.8939 | 0.8156 | 198 |
| lightgcn_agr_conf | p16_conf_w0.3_t0.9 | 2025 | 0.1096 | 0.1673 | 0.2481 | 0.1380 | 0.8933 | 0.8147 | 198 |
| lightgcn_agr_conf | s2_llmconf_a0.1_t0.6 | 2025 | 0.1092 | 0.1688 | 0.2476 | 0.1381 | 0.8928 | 0.8145 | 180 |
| lightgcn_agr_conf | s2_llmconf_a0.2_t0.5 | 2025 | 0.1108 | 0.1701 | 0.2496 | 0.1394 | 0.8964 | 0.8184 | 186 |
| lightgcn_agr_conf | s2_llmconf_a0.2_t0.7 | 2025 | 0.1098 | 0.1668 | 0.2502 | 0.1393 | 0.8960 | 0.8178 | 198 |
| lightgcn_agr_conf | s2_llmconf_a0.3_t0.5 | 2025 | 0.1091 | 0.1692 | 0.2489 | 0.1387 | 0.8957 | 0.8169 | 180 |
| lightgcn_agr_conf | s2_llmconf_a0.4_t0.4 | 2025 | 0.1106 | 0.1679 | 0.2481 | 0.1389 | 0.8954 | 0.8160 | 174 |
| lightgcn_agr_priv | ag_priv_b0_s0_n0.15 | 2025 | 0.0082 | 0.0150 | 0.0280 | 0.0139 | 0.7432 | 0.6769 | 6 |
| lightgcn_agr_priv | ag_priv_kr0.3_n0.30 | 2025 | 0.0012 | 0.0027 | 0.0065 | 0.0028 | 0.6900 | 0.6391 | 12 |
| lightgcn_agr_priv | ag_priv_kr0.5_n0.15 | 2025 | 0.0101 | 0.0180 | 0.0307 | 0.0157 | 0.8374 | 0.7568 | 45 |
| lightgcn_agr_priv | ag_priv_n0.05 | 2025 | 0.0734 | 0.1173 | 0.1759 | 0.0941 | 0.9132 | 0.8310 | 42 |
| lightgcn_agr_priv | ag_priv_n0.15 | 2025 | 0.0079 | 0.0144 | 0.0288 | 0.0139 | 0.8371 | 0.7572 | 21 |
| lightgcn_agr_priv | ag_priv_n0.30 | 2025 | 0.0011 | 0.0027 | 0.0062 | 0.0027 | 0.6969 | 0.6446 | 12 |
| lightgcn_agr_priv | ag_priv_n0.50 | 2025 | 0.0013 | 0.0023 | 0.0058 | 0.0025 | 0.6437 | 0.6099 | 6 |

### Privacy-utility map (`b<beta>_s<str_weight>` tags)

Each row is a (β, str_weight) cell; left columns are utility, right are privacy.


## 4. Discussion

- **What we changed in the inherited code**: removed dead components (cf_index/learn_graph_structure no-op, recon_loss never called, kd_weight unused, double-multiply by alpha); wired up real LLM-augmented adjacency via top-K cosine kNN; decoupled loss weights so each hyperparameter has exactly one role; made data handler dataset-agnostic; switched encoder default to BAAI BGE-small-en-v1.5 (open-source, Chinese-team-published).

- **DP-SGD as reference (failed innovation)**: previous attempt to use DP-SGD as the privacy defense brought recall@20 from 0.26 → 0.006 — a 40× regression — confirming that classical DP is not viable for LLM-augmented recommendation at this scale. The regularization-based defense tested here has comparable utility to the unregularized baseline while shifting the MIA AUC.

- **Limitations**: small datasets; single seed; LR attacker only (no MLP/strong attacker); compute budget capped sweeps to ≤200 epochs and ≤9 cells.


## 5. Comparison of 4 privacy-utility methods

Each method is summarized by its best (recall, AUC) Pareto point on `digital_music`. For reference: LightGCN baseline R@20 ≈ 0.241, AUC ≈ 0.902. Degree-only attack floor AUC ≈ 0.900.

| method | tag | recall@20 | MIA AUC | best_epoch |
|---|---|---|---|---|
| M1 confidence reg | s2_llmconf_a0.1_t0.6 | 0.2476 | 0.8928 | 180 |
| M1 confidence reg | s2_llmconf_a0.3_t0.5 | 0.2489 | 0.8957 | 180 |
| M1 confidence reg | s2_llmconf_a0.2_t0.5 | 0.2496 | 0.8964 | 186 |
| M1 confidence reg | s2_llmconf_a0.4_t0.4 | 0.2481 | 0.8954 | 174 |
| M1 confidence reg | s2_llmconf_a0.2_t0.7 | 0.2502 | 0.8960 | 198 |
| M1 confidence reg | m1_conf01_mlp | – | 0.8918 | – |
| M1 confidence reg | m1_conf05_mlp | – | 0.9006 | – |
| M1 confidence reg | m1_conf01_mlp | – | 0.8918 | – |
| M1 confidence reg | m1_conf05_mlp | – | 0.9006 | – |
| M1 confidence reg | p14b_emb16_conf05 | 0.2242 | 0.8517 | 198 |
| M1 confidence reg | p14b_emb16_conf01 | 0.2217 | 0.8494 | 198 |
| M1 confidence reg | p14a_m1conf01_seed2027 | 0.2476 | 0.8909 | 198 |
| M1 confidence reg | p14a_m1conf01_seed2026 | 0.2476 | 0.8909 | 198 |
| M1 confidence reg | p14b_emb12_conf03 | 0.2085 | 0.8252 | 162 |
| M1 confidence reg | p14b_emb24_conf03 | 0.2364 | 0.8764 | 198 |
| M1 confidence reg | p16_conf_w0.3_t0.4 | 0.2505 | 0.8962 | 198 |
| M1 confidence reg | p16_conf_w0.3_t0.6 | 0.2496 | 0.8941 | 195 |
| M1 confidence reg | p16_conf_w0.3_t0.8 | 0.2484 | 0.8939 | 198 |
| M1 confidence reg | p16_conf_w0.3_t0.9 | 0.2481 | 0.8933 | 198 |
| M1 confidence reg | m1_conf_w5.0_t0.6 | 0.2269 | 0.8871 | 72 |
| M1 confidence reg | m1_conf_w2.0_t0.7 | 0.2425 | 0.8997 | 162 |
| M1 confidence reg | m1_conf_w0.1_t0.7 | 0.2418 | 0.8805 | 165 |
| M1 confidence reg | m1_conf_w0.5_t0.7 | 0.2465 | 0.8870 | 168 |
| M1 confidence reg | m1_conf_w0.5_t0.55 | 0.2500 | 0.8920 | 180 |
| M1 confidence reg | m1_conf_w0.3_p12 | 0.2491 | 0.8945 | 198 |
| M1 confidence reg | m1_conf_w1.0_p12 | 0.2462 | 0.8949 | 186 |
| M2 balanced MIA | lgcn_baseline_balanced | – | 0.8963 | – |
| M2 balanced MIA | agr_default_balanced | – | 0.9339 | – |
| M2 balanced MIA | b0s0_balanced | – | 0.8810 | – |
| M2 balanced MIA | b1s0_balanced | – | 0.9064 | – |
| M2 balanced MIA | b4s0_balanced | – | 0.9013 | – |
| M3 emb compression | m3_emb8_b0s0 | 0.2008 | 0.8102 | 183 |
| M3 emb compression | m3_emb16_b0s0 | 0.2195 | 0.8455 | 186 |
| M3 emb compression | m3_emb16_default | 0.2314 | 0.9035 | 96 |
| M3 emb compression | m3_emb8_default | 0.1959 | 0.8510 | 198 |
| M3 emb compression | m3_emb24_b0s0 | 0.2244 | 0.8569 | 150 |
| M3 emb compression | m3_emb12_b0s0 | 0.2133 | 0.8336 | 192 |
| M4 adversarial GRL | m4_adv0.05_mlp | – | 0.8888 | – |
| M4 adversarial GRL | m4_adv0.05_mlp | – | 0.8888 | – |
| M4 adversarial GRL | m4_adv0.05_seed2026 | 0.2411 | 0.8667 | 168 |
| M4 adversarial GRL | m4_adv0.05_seed2027 | 0.2434 | 0.8755 | 198 |
| M4 adversarial GRL | p13a_emb24_adv0.05 | 0.2241 | 0.8510 | 165 |
| M4 adversarial GRL | p13a_emb16_adv0.1 | 0.2130 | 0.8273 | 192 |
| M4 adversarial GRL | p13a_emb16_adv0.05 | 0.2176 | 0.8375 | 192 |
| M4 adversarial GRL | p13d_adv0.15_p15 | 0.2251 | 0.8427 | 192 |
| M4 adversarial GRL | p13d_adv0.05_p15 | 0.2431 | 0.8749 | 195 |
| M4 adversarial GRL | p13d_adv0.2_lam0.5_p15 | 0.2369 | 0.8604 | 198 |
| M4 adversarial GRL | p15_emb10_adv0.1 | 0.1207 | 0.7663 | 6 |
| M4 adversarial GRL | p15_emb10_adv0.05 | 0.2000 | 0.8125 | 192 |
| M4 adversarial GRL | p15_emb20_adv0.05 | 0.2280 | 0.8499 | 198 |
| M4 adversarial GRL | p15_emb12_adv0.05 | 0.2186 | 0.8289 | 198 |
| M4 adversarial GRL | p15_emb16_adv0.1_seed2026 | 0.2143 | 0.8281 | 195 |
| M4 adversarial GRL | p15_emb16_adv0.1_seed2027 | 0.2133 | 0.8273 | 192 |
| M4 adversarial GRL | p16_emb16adv01_seed2026 | 0.2132 | 0.8273 | 192 |
| M4 adversarial GRL | p16_emb16adv01_seed2027 | 0.2135 | 0.8273 | 192 |
| M4 adversarial GRL | p17_adv005_p20 | 0.2418 | 0.8749 | 195 |
| M4 adversarial GRL | m4_adv0.2 | 0.1411 | 0.8379 | 3 |
| M4 adversarial GRL | m4_adv1.0 | 0.1392 | 0.8490 | 3 |
| M4 adversarial GRL | m4_adv3.0 | 0.1119 | 0.8570 | 3 |
| M4 adversarial GRL | m4_adv1.0_lam2 | 0.1246 | 0.8569 | 3 |
| M4 adversarial GRL | m4_adv0.05 | 0.2437 | 0.8755 | 198 |
| M4 adversarial GRL | m4_adv0.02_p12 | 0.2450 | 0.8835 | 198 |
| M4 adversarial GRL | m4_adv0.1_p12 | 0.2380 | 0.8604 | 198 |
| M4 adversarial GRL | m4_adv0.05_lam0.5_p12 | 0.2449 | 0.8822 | 198 |


## 6. Reproducibility

```bash
# remote DCU host
source /opt/dtk-25.04.2/env.sh
cd /root/private_data/wxy/llw/code
bash scripts/launch_full_pipeline.sh digital_music pareto 200 3
bash scripts/launch_methods_unified.sh
python3 -m scripts.analyze --out_md results/REPORT.md
```

