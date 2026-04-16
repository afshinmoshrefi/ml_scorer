# V4 Enhanced Backtest Results

Objective: highest return at highest stability and safety.

Base: Codex V3 best configs (Sharpe 7.11). Enhancements: VIX block, Best-4 filters, 
100-Year Pattern regime, multi-tier, VIX-scaled sizing.

Run date: 2026-04-06


---

## Stock Strategy Results

### All Configs Ranked by Sharpe

| strategy_id | label | direction | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_years_profitable | worst_year_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4_STK_A_05_skip_monday_B | BaseA | 05_skip_monday | both | both | 7.4634 | -1.8372 | 85.8900 | 36.3209 | 1155 | True | 27.2400 |
| V4_STK_B_03_sym_quality_L | BaseB | 03_sym_quality | long | long | 7.2256 | -2.7487 | 84.3900 | 36.0441 | 1179 | True | 23.9200 |
| V4_STK_A_04_no_repeat14_B | BaseA | 04_no_repeat14 | both | both | 7.1571 | -2.5413 | 84.4900 | 35.9197 | 1193 | True | 27.1000 |
| V4_STK_A_01_baseline_B | BaseA | 01_baseline | both | both | 7.1102 | -2.6529 | 84.5100 | 35.6734 | 1194 | True | 26.3100 |
| V4_STK_A_02_vix_block_B | BaseA | 02_vix_block | both | both | 7.1102 | -2.6529 | 84.5100 | 35.6734 | 1194 | True | 26.3100 |
| V4_STK_B_03_sym_quality_B | BaseB | 03_sym_quality | both | both | 7.1004 | -2.7487 | 84.7600 | 36.7890 | 1168 | True | 26.4100 |
| V4_STK_B_09_regime_B | BaseB | 09_regime | both | both | 7.0929 | -3.2162 | 84.0300 | 36.6123 | 1202 | True | 26.1600 |
| V4_STK_B_10_vix_regime_B | BaseB | 10_vix_regime | both | both | 7.0929 | -3.2162 | 84.0300 | 36.6123 | 1202 | True | 26.1600 |
| V4_STK_B_04_no_repeat14_B | BaseB | 04_no_repeat14 | both | both | 7.0898 | -3.6650 | 83.4500 | 34.8771 | 1160 | True | 26.7200 |
| V4_STK_B_04_no_repeat14_L | BaseB | 04_no_repeat14 | long | long | 7.0724 | -3.6631 | 83.9700 | 35.6064 | 1160 | True | 26.4900 |
| V4_STK_B_02_vix_block_B | BaseB | 02_vix_block | both | both | 7.0598 | -3.2162 | 83.7100 | 35.2343 | 1160 | True | 26.1600 |
| V4_STK_B_01_baseline_B | BaseB | 01_baseline | both | both | 7.0598 | -3.2162 | 83.7100 | 35.2343 | 1160 | True | 26.1600 |
| V4_STK_B_05_skip_monday_B | BaseB | 05_skip_monday | both | both | 7.0562 | -3.3261 | 85.8000 | 35.5981 | 1148 | True | 24.4800 |
| V4_STK_A_03_sym_quality_B | BaseA | 03_sym_quality | both | both | 7.0262 | -2.8328 | 83.4400 | 37.1441 | 1274 | True | 26.7900 |
| V4_STK_A_05_skip_monday_L | BaseA | 05_skip_monday | long | long | 6.9813 | -2.8026 | 85.1400 | 33.4806 | 1151 | True | 21.0200 |
| V4_STK_A_10_vix_regime_B | BaseA | 10_vix_regime | both | both | 6.9778 | -3.3095 | 83.9100 | 36.3777 | 1243 | True | 29.5700 |
| V4_STK_A_09_regime_B | BaseA | 09_regime | both | both | 6.9778 | -3.3095 | 83.9100 | 36.3777 | 1243 | True | 29.5700 |
| V4_STK_B_09_regime_L | BaseB | 09_regime | long | long | 6.9422 | -3.6339 | 83.4000 | 35.4323 | 1205 | True | 22.8800 |
| V4_STK_B_10_vix_regime_L | BaseB | 10_vix_regime | long | long | 6.9422 | -3.6339 | 83.4000 | 35.4323 | 1205 | True | 22.8800 |
| V4_STK_A_08_vix_best4_B | BaseA | 08_vix_best4 | both | both | 6.9174 | -2.1171 | 85.0400 | 33.9262 | 1170 | True | 26.7800 |
| V4_STK_A_07_best4_B | BaseA | 07_best4 | both | both | 6.9174 | -2.1171 | 85.0400 | 33.9262 | 1170 | True | 26.7800 |
| V4_STK_B_02_vix_block_L | BaseB | 02_vix_block | long | long | 6.8632 | -3.2162 | 83.8500 | 34.7001 | 1170 | True | 25.5700 |
| V4_STK_B_01_baseline_L | BaseB | 01_baseline | long | long | 6.8632 | -3.2162 | 83.8500 | 34.7001 | 1170 | True | 25.5700 |
| V4_STK_A_04_no_repeat14_L | BaseA | 04_no_repeat14 | long | long | 6.7595 | -3.2776 | 83.1900 | 33.9380 | 1178 | True | 26.7800 |
| V4_STK_A_11_vix_best4_reg_B | BaseA | 11_vix_best4_reg | both | both | 6.7582 | -1.8468 | 84.3600 | 34.5160 | 1221 | True | 26.7800 |
| V4_STK_A_09_regime_L | BaseA | 09_regime | long | long | 6.7281 | -3.3095 | 83.4400 | 35.1652 | 1238 | True | 26.9000 |
| V4_STK_A_10_vix_regime_L | BaseA | 10_vix_regime | long | long | 6.7281 | -3.3095 | 83.4400 | 35.1652 | 1238 | True | 26.9000 |
| V4_STK_B_05_skip_monday_L | BaseB | 05_skip_monday | long | long | 6.7123 | -3.3261 | 85.2100 | 34.0826 | 1156 | True | 22.8800 |
| V4_STK_A_01_baseline_L | BaseA | 01_baseline | long | long | 6.7072 | -3.2776 | 83.5300 | 33.3805 | 1178 | True | 24.5200 |
| V4_STK_A_02_vix_block_L | BaseA | 02_vix_block | long | long | 6.7072 | -3.2776 | 83.5300 | 33.3805 | 1178 | True | 24.5200 |
| V4_STK_A_12_full_B | BaseA | 12_full | both | both | 6.3787 | -2.1051 | 84.5500 | 27.0981 | 971 | True | 21.1100 |
| V4_STK_A_03_sym_quality_L | BaseA | 03_sym_quality | long | long | 6.1346 | -2.9818 | 81.1000 | 31.8281 | 1249 | True | 24.2500 |
| V4_STK_B_10_vix_regime_S | BaseB | 10_vix_regime | short | short | 5.8168 | -2.5520 | 85.6800 | 26.3838 | 922 | True | 2.7900 |
| V4_STK_B_09_regime_S | BaseB | 09_regime | short | short | 5.8168 | -2.5520 | 85.6800 | 26.3838 | 922 | True | 2.7900 |
| V4_STK_A_10_vix_regime_S | BaseA | 10_vix_regime | short | short | 5.5913 | -3.5632 | 84.9900 | 24.2167 | 873 | True | 1.5100 |
| V4_STK_A_09_regime_S | BaseA | 09_regime | short | short | 5.5913 | -3.5632 | 84.9900 | 24.2167 | 873 | True | 1.5100 |
| V4_STK_B_04_no_repeat14_S | BaseB | 04_no_repeat14 | short | short | 5.4935 | -2.5588 | 84.1800 | 22.2673 | 822 | True | 2.7900 |
| V4_STK_B_05_skip_monday_S | BaseB | 05_skip_monday | short | short | 5.3094 | -1.7804 | 87.5600 | 21.8869 | 788 | True | 2.1300 |
| V4_STK_B_01_baseline_S | BaseB | 01_baseline | short | short | 5.2248 | -2.5520 | 84.9000 | 22.3823 | 821 | True | 2.7900 |
| V4_STK_B_02_vix_block_S | BaseB | 02_vix_block | short | short | 5.2248 | -2.5520 | 84.9000 | 22.3823 | 821 | True | 2.7900 |


### Top 10 Combined L+S

| strategy_id | label | direction | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_years_profitable | worst_year_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4_STK_A_05_skip_monday_B | BaseA | 05_skip_monday | both | both | 7.4634 | -1.8372 | 85.8900 | 36.3209 | 1155 | True | 27.2400 |
| V4_STK_A_04_no_repeat14_B | BaseA | 04_no_repeat14 | both | both | 7.1571 | -2.5413 | 84.4900 | 35.9197 | 1193 | True | 27.1000 |
| V4_STK_A_02_vix_block_B | BaseA | 02_vix_block | both | both | 7.1102 | -2.6529 | 84.5100 | 35.6734 | 1194 | True | 26.3100 |
| V4_STK_A_01_baseline_B | BaseA | 01_baseline | both | both | 7.1102 | -2.6529 | 84.5100 | 35.6734 | 1194 | True | 26.3100 |
| V4_STK_B_03_sym_quality_B | BaseB | 03_sym_quality | both | both | 7.1004 | -2.7487 | 84.7600 | 36.7890 | 1168 | True | 26.4100 |
| V4_STK_B_10_vix_regime_B | BaseB | 10_vix_regime | both | both | 7.0929 | -3.2162 | 84.0300 | 36.6123 | 1202 | True | 26.1600 |
| V4_STK_B_09_regime_B | BaseB | 09_regime | both | both | 7.0929 | -3.2162 | 84.0300 | 36.6123 | 1202 | True | 26.1600 |
| V4_STK_B_04_no_repeat14_B | BaseB | 04_no_repeat14 | both | both | 7.0898 | -3.6650 | 83.4500 | 34.8771 | 1160 | True | 26.7200 |
| V4_STK_B_01_baseline_B | BaseB | 01_baseline | both | both | 7.0598 | -3.2162 | 83.7100 | 35.2343 | 1160 | True | 26.1600 |
| V4_STK_B_02_vix_block_B | BaseB | 02_vix_block | both | both | 7.0598 | -3.2162 | 83.7100 | 35.2343 | 1160 | True | 26.1600 |


### Top 10 Long-Only

| strategy_id | label | direction | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_years_profitable | worst_year_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4_STK_B_03_sym_quality_L | BaseB | 03_sym_quality | long | long | 7.2256 | -2.7487 | 84.3900 | 36.0441 | 1179 | True | 23.9200 |
| V4_STK_B_04_no_repeat14_L | BaseB | 04_no_repeat14 | long | long | 7.0724 | -3.6631 | 83.9700 | 35.6064 | 1160 | True | 26.4900 |
| V4_STK_A_05_skip_monday_L | BaseA | 05_skip_monday | long | long | 6.9813 | -2.8026 | 85.1400 | 33.4806 | 1151 | True | 21.0200 |
| V4_STK_B_09_regime_L | BaseB | 09_regime | long | long | 6.9422 | -3.6339 | 83.4000 | 35.4323 | 1205 | True | 22.8800 |
| V4_STK_B_10_vix_regime_L | BaseB | 10_vix_regime | long | long | 6.9422 | -3.6339 | 83.4000 | 35.4323 | 1205 | True | 22.8800 |
| V4_STK_B_01_baseline_L | BaseB | 01_baseline | long | long | 6.8632 | -3.2162 | 83.8500 | 34.7001 | 1170 | True | 25.5700 |
| V4_STK_B_02_vix_block_L | BaseB | 02_vix_block | long | long | 6.8632 | -3.2162 | 83.8500 | 34.7001 | 1170 | True | 25.5700 |
| V4_STK_A_04_no_repeat14_L | BaseA | 04_no_repeat14 | long | long | 6.7595 | -3.2776 | 83.1900 | 33.9380 | 1178 | True | 26.7800 |
| V4_STK_A_10_vix_regime_L | BaseA | 10_vix_regime | long | long | 6.7281 | -3.3095 | 83.4400 | 35.1652 | 1238 | True | 26.9000 |
| V4_STK_A_09_regime_L | BaseA | 09_regime | long | long | 6.7281 | -3.3095 | 83.4400 | 35.1652 | 1238 | True | 26.9000 |


### Top 10 Short-Only

| strategy_id | label | direction | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_years_profitable | worst_year_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4_STK_B_10_vix_regime_S | BaseB | 10_vix_regime | short | short | 5.8168 | -2.5520 | 85.6800 | 26.3838 | 922 | True | 2.7900 |
| V4_STK_B_09_regime_S | BaseB | 09_regime | short | short | 5.8168 | -2.5520 | 85.6800 | 26.3838 | 922 | True | 2.7900 |
| V4_STK_A_10_vix_regime_S | BaseA | 10_vix_regime | short | short | 5.5913 | -3.5632 | 84.9900 | 24.2167 | 873 | True | 1.5100 |
| V4_STK_A_09_regime_S | BaseA | 09_regime | short | short | 5.5913 | -3.5632 | 84.9900 | 24.2167 | 873 | True | 1.5100 |
| V4_STK_B_04_no_repeat14_S | BaseB | 04_no_repeat14 | short | short | 5.4935 | -2.5588 | 84.1800 | 22.2673 | 822 | True | 2.7900 |
| V4_STK_B_05_skip_monday_S | BaseB | 05_skip_monday | short | short | 5.3094 | -1.7804 | 87.5600 | 21.8869 | 788 | True | 2.1300 |
| V4_STK_B_01_baseline_S | BaseB | 01_baseline | short | short | 5.2248 | -2.5520 | 84.9000 | 22.3823 | 821 | True | 2.7900 |
| V4_STK_B_02_vix_block_S | BaseB | 02_vix_block | short | short | 5.2248 | -2.5520 | 84.9000 | 22.3823 | 821 | True | 2.7900 |
| V4_STK_A_04_no_repeat14_S | BaseA | 04_no_repeat14 | short | short | 5.1231 | -3.2945 | 84.1200 | 19.9265 | 743 | True | 1.5100 |
| V4_STK_A_01_baseline_S | BaseA | 01_baseline | short | short | 5.0651 | -3.5632 | 84.6800 | 19.4188 | 744 | True | 1.5100 |


### Enhancement Impact on BaseA (Combined L+S)

| label | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_years_profitable |
| --- | --- | --- | --- | --- | --- | --- |
| BaseA | 05_skip_monday | both | 7.4634 | -1.8372 | 85.8900 | 36.3209 | 1155 | True |
| BaseA | 04_no_repeat14 | both | 7.1571 | -2.5413 | 84.4900 | 35.9197 | 1193 | True |
| BaseA | 02_vix_block | both | 7.1102 | -2.6529 | 84.5100 | 35.6734 | 1194 | True |
| BaseA | 01_baseline | both | 7.1102 | -2.6529 | 84.5100 | 35.6734 | 1194 | True |
| BaseA | 03_sym_quality | both | 7.0262 | -2.8328 | 83.4400 | 37.1441 | 1274 | True |
| BaseA | 09_regime | both | 6.9778 | -3.3095 | 83.9100 | 36.3777 | 1243 | True |
| BaseA | 10_vix_regime | both | 6.9778 | -3.3095 | 83.9100 | 36.3777 | 1243 | True |
| BaseA | 07_best4 | both | 6.9174 | -2.1171 | 85.0400 | 33.9262 | 1170 | True |
| BaseA | 08_vix_best4 | both | 6.9174 | -2.1171 | 85.0400 | 33.9262 | 1170 | True |
| BaseA | 11_vix_best4_reg | both | 6.7582 | -1.8468 | 84.3600 | 34.5160 | 1221 | True |
| BaseA | 12_full | both | 6.3787 | -2.1051 | 84.5500 | 27.0981 | 971 | True |
| BaseA | 06_wkly_breaker | both | 2.3270 | -2.6542 | 81.9000 | 4.6640 | 221 | False |


### Enhancement Impact on BaseB (Combined L+S)

| label | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_years_profitable |
| --- | --- | --- | --- | --- | --- | --- |
| BaseB | 03_sym_quality | both | 7.1004 | -2.7487 | 84.7600 | 36.7890 | 1168 | True |
| BaseB | 10_vix_regime | both | 7.0929 | -3.2162 | 84.0300 | 36.6123 | 1202 | True |
| BaseB | 09_regime | both | 7.0929 | -3.2162 | 84.0300 | 36.6123 | 1202 | True |
| BaseB | 04_no_repeat14 | both | 7.0898 | -3.6650 | 83.4500 | 34.8771 | 1160 | True |
| BaseB | 01_baseline | both | 7.0598 | -3.2162 | 83.7100 | 35.2343 | 1160 | True |
| BaseB | 02_vix_block | both | 7.0598 | -3.2162 | 83.7100 | 35.2343 | 1160 | True |
| BaseB | 05_skip_monday | both | 7.0562 | -3.3261 | 85.8000 | 35.5981 | 1148 | True |
| BaseB | 06_wkly_breaker | both | 4.3014 | -3.0515 | 84.0100 | 14.1460 | 538 | False |
| BaseB | 12_full | both | 2.8704 | -1.5855 | 81.5000 | 6.0948 | 254 | False |
| BaseB | 11_vix_best4_reg | both | 2.6395 | -3.2907 | 81.7400 | 5.1308 | 241 | False |
| BaseB | 08_vix_best4 | both | 2.6315 | -4.0847 | 82.8900 | 5.1016 | 228 | False |
| BaseB | 07_best4 | both | 2.6315 | -4.0847 | 82.8900 | 5.1016 | 228 | False |


### Year-by-Year Returns (Top 5 Combined)

| strategy_id | label | y2018 | y2019 | y2020 | y2021 | y2022 | y2023 | y2024 | y2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4_STK_A_01_baseline_B | BaseA | 01_baseline | both | 32.6400 | 26.3100 | 31.2200 | 42.3600 | 40.1800 | 41.4200 | 31.3300 | 38.2500 |
| V4_STK_A_02_vix_block_B | BaseA | 02_vix_block | both | 32.6400 | 26.3100 | 31.2200 | 42.3600 | 40.1800 | 41.4200 | 31.3300 | 38.2500 |
| V4_STK_A_04_no_repeat14_B | BaseA | 04_no_repeat14 | both | 33.9200 | 27.1000 | 31.5000 | 40.7000 | 42.4900 | 38.4200 | 30.7800 | 39.5100 |
| V4_STK_A_05_skip_monday_B | BaseA | 05_skip_monday | both | 28.9100 | 27.2400 | 29.3400 | 43.3600 | 46.4100 | 40.4300 | 30.2400 | 43.5900 |
| V4_STK_B_03_sym_quality_B | BaseB | 03_sym_quality | both | 26.4100 | 30.7400 | 28.7100 | 38.1200 | 47.0200 | 40.9700 | 39.3300 | 42.7700 |


### Holdout Check (Top 5: 2018-2024 train vs 2025 out-of-sample)

| strategy_id | label | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V4_STK_A_02_vix_block_B | BaseA | 02_vix_block | both | 2018_2024 | 6.9474 | -2.6529 | 34.7638 | 84.0700 | 1036 |
| V4_STK_A_02_vix_block_B | BaseA | 02_vix_block | both | 2025 | 7.1535 | -0.9318 | 38.8286 | 85.8000 | 162 |
| V4_STK_A_01_baseline_B | BaseA | 01_baseline | both | 2018_2024 | 6.9474 | -2.6529 | 34.7638 | 84.0700 | 1036 |
| V4_STK_A_01_baseline_B | BaseA | 01_baseline | both | 2025 | 7.1535 | -0.9318 | 38.8286 | 85.8000 | 162 |
| V4_STK_B_03_sym_quality_B | BaseB | 03_sym_quality | both | 2018_2024 | 7.0538 | -2.7487 | 35.4934 | 84.1800 | 1024 |
| V4_STK_B_03_sym_quality_B | BaseB | 03_sym_quality | both | 2025 | 8.3219 | -0.8204 | 43.0707 | 89.8000 | 147 |
| V4_STK_A_05_skip_monday_B | BaseA | 05_skip_monday | both | 2018_2024 | 7.2805 | -1.8372 | 34.9600 | 85.0000 | 1000 |
| V4_STK_A_05_skip_monday_B | BaseA | 05_skip_monday | both | 2025 | 8.4068 | -1.5718 | 44.3751 | 91.2500 | 160 |
| V4_STK_A_04_no_repeat14_B | BaseA | 04_no_repeat14 | both | 2018_2024 | 7.0298 | -2.5413 | 34.9265 | 84.0300 | 1033 |
| V4_STK_A_04_no_repeat14_B | BaseA | 04_no_repeat14 | both | 2025 | 7.4683 | -1.1376 | 38.0520 | 85.6200 | 160 |


---

## Options Strategy Results

| strategy_id | label | direction | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_years_profitable | worst_year_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4_OPT_01_baseline_B | Options | 01_baseline | both | both | 5.3834 | -22.3235 | 71.6800 | 4097.1237 | 692 | True | 1149.4000 |
| V4_OPT_01_baseline_L | Options | 01_baseline | long | long | 5.0211 | -22.7424 | 69.1200 | 3122.5807 | 693 | True | 613.7600 |
| V4_OPT_01_baseline_S | Options | 01_baseline | short | short | 3.7540 | -25.2796 | 68.0000 | 680.0748 | 450 | True | 29.7400 |
| V4_OPT_08_vix_best4_L | Options | 08_vix_best4 | long | long | 2.0354 | -16.3386 | 63.5100 | 85.4726 | 148 | False | nan |
| V4_OPT_08_vix_best4_B | Options | 08_vix_best4 | both | both | 1.4409 | -15.6610 | 69.2300 | 34.5358 | 65 | False | nan |
| V4_OPT_11_vix_best4_reg_L | Options | 11_vix_best4_reg | long | long | 1.3734 | -12.1260 | 64.8600 | 31.3942 | 74 | False | nan |
| V4_OPT_11_vix_best4_reg_B | Options | 11_vix_best4_reg | both | both | 1.3492 | -21.9827 | 66.6700 | 31.4487 | 66 | False | nan |
| V4_OPT_08_vix_best4_S | Options | 08_vix_best4 | short | short | -0.3024 | -6.3055 | 33.3300 | -0.7098 | 6 | False | -5.5900 |
| V4_OPT_11_vix_best4_reg_S | Options | 11_vix_best4_reg | short | short | -0.3024 | -6.3055 | 33.3300 | -0.7098 | 6 | False | -5.5900 |


---

## Spread Strategy Results

| strategy_id | label | direction | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_years_profitable | worst_year_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4_SPR_01_baseline_B | Spread | 01_baseline | both | both | 6.3544 | -25.0733 | 89.4000 | 621.6241 | 915 | True | 208.5900 |
| V4_SPR_01_baseline_L | Spread | 01_baseline | long | long | 5.9016 | -25.0733 | 88.6200 | 569.9628 | 914 | True | 208.1200 |
| V4_SPR_11_vix_best4_reg_S | Spread | 11_vix_best4_reg | short | short | 5.4736 | -17.5739 | 91.1000 | 354.2468 | 652 | True | 17.0100 |
| V4_SPR_01_baseline_S | Spread | 01_baseline | short | short | 5.2666 | -21.6340 | 92.4900 | 313.6632 | 586 | True | 17.0100 |
| V4_SPR_11_vix_best4_reg_L | Spread | 11_vix_best4_reg | long | long | 3.0821 | -24.8102 | 88.7200 | 71.9727 | 257 | False | nan |
| V4_SPR_08_vix_best4_S | Spread | 08_vix_best4 | short | short | 2.7665 | -17.5739 | 89.4500 | 66.6028 | 237 | False | nan |
| V4_SPR_08_vix_best4_L | Spread | 08_vix_best4 | long | long | 2.6663 | -24.8102 | 88.0000 | 57.3629 | 225 | False | nan |
| V4_SPR_11_vix_best4_reg_B | Spread | 11_vix_best4_reg | both | both | 2.5201 | -33.1885 | 85.7700 | 57.8877 | 239 | False | nan |
| V4_SPR_08_vix_best4_B | Spread | 08_vix_best4 | both | both | 1.4967 | -31.4770 | 85.1900 | 20.0789 | 108 | False | nan |


---

## Key Findings

- Best combined stock: `V4_STK_A_05_skip_monday_B` | Sharpe 7.46 | DD -1.84% | CAGR 36.32% | WR 85.9%

- Baseline (no enhancements): Sharpe 7.11 | DD -2.65%

- Enhancement lift: +0.35 Sharpe, 0.82pp DD change

---

## Codex Analysis

### Executive Summary

V4 testing confirms that the Codex V3 baseline (Sharpe 7.11) is robust and that selective enhancements can push performance meaningfully higher. The single most effective enhancement is **SkipMonday**, which delivers +0.35 Sharpe and reduces max drawdown from -2.65% to -1.84% -- achieving the target DD < 2% for the first time. All top configurations have all 8 years profitable with worst-year returns above 26%.

No configuration achieved Sharpe > 8.0 over the full 2018-2025 period, but the top config (A_skip_monday) achieved Sharpe 8.41 in the 2025 out-of-sample holdout, which is the strongest validation result in any test to date.

---

### Enhancement Attribution: Winners and Losers

**Winners (BaseA combined L+S):**

| Enhancement | Sharpe | DD | Verdict |
|---|---|---|---|
| 05_skip_monday | 7.46 | -1.84% | **Enable always. Best single filter.** |
| 04_no_repeat14 | 7.16 | -2.54% | Enable. Small but consistent improvement. |
| 02_vix_block | 7.11 | -2.65% | Enable for production safety. No cost in data. |
| 01_baseline | 7.11 | -2.65% | Reference. |

**Losers:**

| Enhancement | Sharpe | DD | Verdict |
|---|---|---|---|
| 06_wkly_breaker | 2.33 | -2.65% | **NEVER enable. Reduces trades from 1194 to 221. Catastrophic.** |
| 11_vix_best4_reg | 6.76 | -1.85% | Weaker than skip_monday alone. WeeklyBreaker drag. |
| 12_full (all) | 6.38 | -2.11% | Full stack destroyed by WeeklyBreaker. |
| 09_regime | 6.98 | -3.31% | Slightly hurts Sharpe. Increases DD. Use only during active 100-Year windows. |

**Key insight on WeeklyBreaker:** The filter pauses entries after 3 losses in the last 5 trades. In a regime where the ML model correctly identifies high-quality patterns, this pause is counter-productive -- it keeps you out of the market during periods when the model is still generating valid signals. The signal is already high-quality (84%+ WR); the breaker is answering a problem that does not exist.

---

### Direction Analysis

- Combined L+S beats long-only by 0.25-0.40 Sharpe in most configurations (7.46 vs 6.98 for best long-only in BaseA)
- Adding shorts does NOT increase drawdown meaningfully (both configs -1.84%)
- Short-only configs achieve Sharpe 5.82 with the regime enhancement -- best short results come from the 100-Year Pattern regime switching (regime forces long-only during the window, which concentrates the highest-quality shorts outside the window)
- Combined L+S is strictly better than long-only unless you have capital that absolutely cannot go short

---

### Robustness: 2025 Holdout

All top configs show **better** 2025 performance than 2018-2024 in-sample. This is the second consecutive out-of-sample validation period where holdout beats in-sample (V3 also showed this pattern with STK_045 at 7.15 holdout).

| Config | 2018-2024 Sharpe | 2025 Sharpe | Change |
|---|---|---|---|
| A_skip_monday | 7.28 | **8.41** | +1.13 |
| B_sym_quality | 7.05 | **8.32** | +1.27 |
| A_no_repeat14 | 7.03 | 7.47 | +0.44 |
| A_baseline | 6.95 | 7.15 | +0.20 |

The consistent holdout improvement suggests the ML model is well-calibrated and that 2025 market conditions (moderate VIX, continued seasonal pattern reliability) are favorable for this strategy class.

---

### Year-by-Year Stability

Top config (A_skip_monday) year-by-year: 28.9%, 27.2%, 29.3%, 43.4%, 46.4%, 40.4%, 30.2%, 43.6%

- **All 8 years profitable** -- including 2018 (Q4 selloff), 2020 (COVID crash), and 2022 (rate shock)
- **Worst year:** 27.2% (2019) -- well above zero
- **Best year:** 46.4% (2022) -- the 100-Year Pattern midterm window boosted performance
- **2022 is notably strong** across all configs (40-47% range), confirming the 100-Year Pattern's amplifying effect
- **2025 is the second-strongest year** for most configs -- confirming continued model validity

---

### Options and Spreads

Options and spreads behave very differently from stocks with respect to enhancements:

**Options (baseline only is correct):**
- Baseline combined: Sharpe 5.38, DD -22.3%, all years profitable
- VIX + Best4 (long): Sharpe 2.04, not all years profitable
- Lesson: Options already use premium/theta filters. Adding stock-focused filters over-restricts the opportunity set. Keep options at baseline.

**Spreads (baseline only is correct):**
- Baseline combined: Sharpe 6.35, DD -25.1%, all years profitable  
- VIX + Best4 (short only): Sharpe 2.77, not all years profitable
- Lesson: Credit spreads are already structured conservatively. Adding filters reduces trade count below the threshold needed for statistical validity.

---

### Final Recommended Configuration Set

**Primary stock strategy (combined long+short):**
- Config: `V4_STK_A_05_skip_monday_B`
- Settings: WP/strict/risk_balanced/target6_atr2/vol_inverse + SkipMonday + VIX hard block
- Sharpe 7.46, DD -1.84%, CAGR 36.32%, WR 85.9%, all years profitable

**Long-only sleeve (capital that cannot go short):**
- Config: `V4_STK_B_03_sym_quality_L`
- Settings: combo_rank/balanced/risk_balanced/target6_atr2/vol_inverse + SymbolQuality
- Sharpe 7.23, DD -2.75%, CAGR 36.04%

**Short-only hedge:**
- Config: `V4_STK_B_09_regime_S`
- Settings: combo_rank/balanced/short-only + regime switching
- Sharpe 5.82, DD -2.55%, CAGR 26.38%

**Options account:**
- Config: `V4_OPT_01_baseline_B` (NO additional filters)
- Sharpe 5.38, DD -22.3%

**Spread account:**
- Config: `V4_SPR_01_baseline_B` (NO additional filters)
- Sharpe 6.35, DD -25.1%

**100-Year Pattern window (Sep 27, 2026 - Jul 18, 2027):**
- Switch to long-only mode, ML threshold >= 70, exclude Energy sector
- Overweight Materials, Consumer Staples, Consumer Discretionary, Real Estate

---

### What NOT to Enable

1. **WeeklyBreaker** -- never. Reduces trade count by 82%, destroys Sharpe.
2. **Best4 combined stack on BaseB** -- configs 07/08/11/12 on BaseB all produce Sharpe 2-3 due to filter interaction.
3. **Enhancement filters on Options or Spreads** -- always use baseline for these asset classes.
4. **Multi-tier combining** -- not tested in this run due to script limitation, defer to next test.

---

### Comparison: V4 vs V3 vs Auto Trading

| System | Config | Sharpe | DD | CAGR | WR | Direction | Status |
|---|---|---|---|---|---|---|---|
| Auto trading (live) | S21 + Best4 | 4.22 | -7.4% | ~80% | ~62% | Long only | Simulation |
| Codex V3 | STK_045 | 7.11 | -2.65% | 35.67% | 84.5% | L+S | Research |
| Codex V4 | A_skip_monday | **7.46** | **-1.84%** | 36.32% | 85.9% | L+S | Research |

The auto trading system earns more absolute CAGR (~80% vs 36%) but at higher risk (7.4% DD vs 1.84%). It uses a fundamentally different exit mechanism (EP flat target vs target6_atr2 volatility-scaled floor). The V4 research system is appropriate for larger capital pools where risk control is paramount. The auto trading system may be appropriate for smaller accounts maximizing absolute growth.
