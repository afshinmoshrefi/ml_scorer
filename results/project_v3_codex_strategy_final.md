# V3 Strategy Backtest Results -- Independent Assessment

## Data Exploration Summary
- Rows analyzed: 11,004,843
- Symbols: 475
- Date range: 2018-01-02 to 2025-12-31
- Base win rate: 67.76%
- Trades removed by earnings filter: 23.19%
- Correlation between predicted_return and actual_return: 0.2416
- Correlation between predicted_mfe and actual_mfe: 0.4595
- Actual return distribution: mean 1.94%, median 1.98%, 5th pct -9.07%, 95th pct 12.74%
- Actual MFE distribution: mean 5.67%, median 4.49%

### By Direction
| direction | trades | win_rate | avg_return | avg_predicted_return |
| --- | --- | --- | --- | --- |
| l | 8039032 | 70.0973 | 2.2560 | 1.5589 |
| s | 2965811 | 61.4063 | 1.0729 | 0.5138 |

### Win Rate by ML Score Decile
| ml_decile | trades | win_rate | avg_return |
| --- | --- | --- | --- |
| 1.0000 | 1105747.0000 | 55.4517 | 0.2801 |
| 2.0000 | 1100016.0000 | 55.8103 | 0.3430 |
| 3.0000 | 1101279.0000 | 58.0019 | 0.4030 |
| 4.0000 | 1100294.0000 | 61.4197 | 0.8140 |
| 5.0000 | 1100720.0000 | 66.4424 | 1.3059 |
| 6.0000 | 1100631.0000 | 70.0501 | 1.7874 |
| 7.0000 | 1100531.0000 | 73.1866 | 2.3186 |
| 8.0000 | 1100683.0000 | 75.8951 | 2.7816 |
| 9.0000 | 1101941.0000 | 78.9736 | 3.5358 |
| 10.0000 | 1093001.0000 | 82.4625 | 5.8349 |

### Win Rate by Win Probability Decile
| wp_decile | trades | win_rate | avg_return |
| --- | --- | --- | --- |
| 1.0000 | 1650681.0000 | 55.1810 | 0.2174 |
| 2.0000 | 1100610.0000 | 57.4748 | 0.4704 |
| 3.0000 | 1100567.0000 | 58.7800 | 0.5372 |
| 4.0000 | 1100499.0000 | 64.1988 | 1.0899 |
| 5.0000 | 550247.0000 | 67.6743 | 1.4318 |
| 6.0000 | 1650608.0000 | 70.8334 | 1.9287 |
| 7.0000 | 1100388.0000 | 74.5962 | 2.5327 |
| 8.0000 | 550332.0000 | 76.4813 | 2.9083 |
| 9.0000 | 1100443.0000 | 78.9549 | 3.5299 |
| 10.0000 | 1100468.0000 | 82.4488 | 5.8227 |

### Performance by Year
| year | trades | win_rate | avg_return | avg_mfe |
| --- | --- | --- | --- | --- |
| 2018.0000 | 1398325.0000 | 71.7079 | 1.8758 | 5.2446 |
| 2019.0000 | 1417598.0000 | 64.1701 | 1.2671 | 4.4749 |
| 2020.0000 | 1153877.0000 | 62.3986 | 1.4004 | 6.7963 |
| 2021.0000 | 1413828.0000 | 68.8986 | 2.0826 | 5.4663 |
| 2022.0000 | 1399367.0000 | 81.6349 | 4.1796 | 8.0221 |
| 2023.0000 | 1413136.0000 | 65.0107 | 1.7372 | 5.2552 |
| 2024.0000 | 1415553.0000 | 65.8827 | 1.7984 | 5.1814 |
| 2025.0000 | 1393159.0000 | 61.4559 | 1.0691 | 5.1683 |

### Performance by Sector
| sector | trades | win_rate | avg_return |
| --- | --- | --- | --- |
| Unknown | 658863 | 69.6920 | 2.5736 |
| Information Technology | 1341758 | 67.8682 | 2.3791 |
| Consumer Discretionary | 1054722 | 67.6244 | 2.2523 |
| Communication Services | 350454 | 69.5498 | 2.2194 |
| Energy | 378459 | 66.2907 | 2.0879 |
| Materials | 461208 | 66.5374 | 2.0355 |
| Industrials | 1721578 | 67.5191 | 1.8675 |
| Financials | 1480631 | 68.5862 | 1.8156 |
| Health Care | 1229496 | 67.3177 | 1.8060 |
| Real Estate | 748059 | 66.8890 | 1.5932 |
| Consumer Staples | 772998 | 67.4483 | 1.5148 |

### Performance by Holding Days
| holding_days | trades | win_rate | avg_return |
| --- | --- | --- | --- |
| 10.0000 | 490464.0000 | 66.9847 | 1.3774 |
| 11.0000 | 494547.0000 | 67.3046 | 1.4498 |
| 12.0000 | 498884.0000 | 67.4520 | 1.5125 |
| 13.0000 | 503020.0000 | 68.0170 | 1.5627 |
| 14.0000 | 505703.0000 | 68.3886 | 1.6116 |
| 15.0000 | 510371.0000 | 67.6806 | 1.6784 |
| 16.0000 | 512783.0000 | 67.5110 | 1.7307 |
| 17.0000 | 514483.0000 | 67.5082 | 1.7804 |
| 18.0000 | 517886.0000 | 67.5257 | 1.8443 |
| 19.0000 | 523561.0000 | 67.5448 | 1.8789 |
| 20.0000 | 529093.0000 | 67.9595 | 1.9208 |
| 21.0000 | 531978.0000 | 68.2722 | 1.9904 |
| 22.0000 | 536007.0000 | 67.6958 | 2.0393 |
| 23.0000 | 537947.0000 | 67.7085 | 2.1074 |
| 24.0000 | 539656.0000 | 67.6094 | 2.1405 |
| 25.0000 | 541296.0000 | 67.5569 | 2.1823 |
| 26.0000 | 542512.0000 | 67.6385 | 2.2192 |
| 27.0000 | 543298.0000 | 68.1727 | 2.2757 |
| 28.0000 | 542380.0000 | 68.4789 | 2.3463 |
| 29.0000 | 543563.0000 | 67.9009 | 2.3784 |
| 30.0000 | 545411.0000 | 67.8334 | 2.4336 |

## Backtesting Methodology
- Starting capital: stocks $100,000, options $10,000, spreads $25,000
- Combined runs include both `l` and `s`; separate long-only and short-only evaluations use the same strategy grid on direction-filtered subsets
- Earnings filter excluded any trade with a symbol-level earnings date between entry and modeled exit
- Equity is tracked on an event basis and open positions are carried at cost until exit, so drawdown is based on realized equity changes rather than intraday mark-to-market noise
- Stock exits tested: hold to scheduled close, +4% target with a 2% trailing approximation, and +6% target with a 2x ATR downside floor
- Options use a simplified ATM call premium model with 2.5%-3.0% starting premium, capped downside at -100%, and theta drag scaled by holding period
- Spreads use simplified bull call debit and bull put credit payoff curves defined directly in percentage return space

## Stock Strategy Results
### Combined Long + Short
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STK_045 | rank=win_probability; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 7.1102 | -2.6529 | 84.5059 | 35.6734 | 1194 | True |
| STK_063 | rank=combo_rank; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 7.0598 | -3.2162 | 83.7069 | 35.2343 | 1160 | True |
| STK_072 | rank=combo_rank; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 7.0486 | -3.2162 | 84.3310 | 34.9531 | 1136 | True |
| STK_036 | rank=win_probability; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.9959 | -2.9649 | 84.0592 | 36.2645 | 1217 | True |
| STK_009 | rank=ml_score; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.9405 | -2.6663 | 84.3990 | 36.2343 | 1173 | True |
| STK_018 | rank=ml_score; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.9392 | -3.0756 | 84.6696 | 34.6398 | 1135 | True |
| STK_054 | rank=win_probability; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.8839 | -2.0164 | 84.6432 | 33.6428 | 1107 | True |
| STK_027 | rank=ml_score; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.7187 | -2.0698 | 84.7328 | 32.5665 | 1048 | True |
| STK_057 | rank=combo_rank; threshold=balanced; concentration=diversified; exit=target6_atr2; size=equal | 6.6791 | -6.0055 | 85.3476 | 64.9854 | 935 | True |
| STK_081 | rank=combo_rank; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.6014 | -3.2162 | 83.9847 | 31.8346 | 1049 | True |
| STK_012 | rank=ml_score; threshold=strict; concentration=diversified; exit=target6_atr2; size=equal | 6.5175 | -5.2905 | 85.4031 | 62.2277 | 918 | True |
| STK_003 | rank=ml_score; threshold=balanced; concentration=diversified; exit=target6_atr2; size=equal | 6.3984 | -6.7468 | 84.9680 | 62.7787 | 938 | True |
| STK_039 | rank=win_probability; threshold=strict; concentration=diversified; exit=target6_atr2; size=equal | 6.3563 | -4.3974 | 84.3979 | 61.9360 | 955 | True |
| STK_030 | rank=win_probability; threshold=balanced; concentration=diversified; exit=target6_atr2; size=equal | 6.3081 | -4.7392 | 83.1622 | 61.1209 | 974 | True |
| STK_066 | rank=combo_rank; threshold=strict; concentration=diversified; exit=target6_atr2; size=equal | 6.3024 | -5.6174 | 84.9616 | 61.5395 | 911 | True |
| STK_021 | rank=ml_score; threshold=elite; concentration=diversified; exit=target6_atr2; size=equal | 6.1812 | -3.9060 | 85.3114 | 58.0632 | 851 | True |
| STK_075 | rank=combo_rank; threshold=elite; concentration=diversified; exit=target6_atr2; size=equal | 6.1745 | -5.6174 | 84.9704 | 56.8012 | 845 | True |
| STK_048 | rank=win_probability; threshold=elite; concentration=diversified; exit=target6_atr2; size=equal | 6.0740 | -4.7370 | 83.4842 | 56.3390 | 884 | True |
| STK_042 | rank=win_probability; threshold=strict; concentration=focused; exit=target6_atr2; size=confidence | 5.6413 | -4.6520 | 84.6026 | 70.0847 | 604 | True |
| STK_069 | rank=combo_rank; threshold=strict; concentration=focused; exit=target6_atr2; size=confidence | 5.6047 | -7.4661 | 86.6432 | 68.1199 | 569 | True |
| STK_060 | rank=combo_rank; threshold=balanced; concentration=focused; exit=target6_atr2; size=confidence | 5.5425 | -8.1442 | 85.6655 | 68.6545 | 586 | True |
| STK_043 | rank=win_probability; threshold=strict; concentration=risk_balanced; exit=hold; size=vol_inverse | 5.3704 | -5.7450 | 82.2446 | 55.1202 | 1194 | True |
| STK_024 | rank=ml_score; threshold=elite; concentration=focused; exit=target6_atr2; size=confidence | 5.3423 | -5.7767 | 86.2454 | 58.0846 | 538 | True |
| STK_033 | rank=win_probability; threshold=balanced; concentration=focused; exit=target6_atr2; size=confidence | 5.3180 | -7.2119 | 83.2248 | 69.5948 | 614 | True |
| STK_015 | rank=ml_score; threshold=strict; concentration=focused; exit=target6_atr2; size=confidence | 5.2933 | -6.9430 | 86.0781 | 66.9623 | 589 | True |
| STK_006 | rank=ml_score; threshold=balanced; concentration=focused; exit=target6_atr2; size=confidence | 5.2792 | -7.4840 | 85.5172 | 65.8566 | 580 | True |
| STK_078 | rank=combo_rank; threshold=elite; concentration=focused; exit=target6_atr2; size=confidence | 5.1978 | -6.9212 | 84.7866 | 55.5274 | 539 | True |
| STK_034 | rank=win_probability; threshold=balanced; concentration=risk_balanced; exit=hold; size=vol_inverse | 5.1042 | -6.7029 | 81.1011 | 55.5941 | 1217 | True |
| STK_051 | rank=win_probability; threshold=elite; concentration=focused; exit=target6_atr2; size=confidence | 4.9456 | -3.9720 | 84.3806 | 56.0286 | 557 | True |
| STK_016 | rank=ml_score; threshold=strict; concentration=risk_balanced; exit=hold; size=vol_inverse | 4.8892 | -5.4848 | 81.5859 | 64.7934 | 1135 | True |
| STK_052 | rank=win_probability; threshold=elite; concentration=risk_balanced; exit=hold; size=vol_inverse | 4.8515 | -7.4195 | 81.8428 | 50.0601 | 1107 | True |
| STK_070 | rank=combo_rank; threshold=strict; concentration=risk_balanced; exit=hold; size=vol_inverse | 4.8017 | -5.4848 | 80.7218 | 68.2876 | 1136 | True |
| STK_007 | rank=ml_score; threshold=balanced; concentration=risk_balanced; exit=hold; size=vol_inverse | 4.7827 | -6.8524 | 81.1594 | 69.4794 | 1173 | True |
| STK_044 | rank=win_probability; threshold=strict; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 4.7161 | -8.9669 | 89.0285 | 21.5419 | 1194 | True |
| STK_061 | rank=combo_rank; threshold=balanced; concentration=risk_balanced; exit=hold; size=vol_inverse | 4.6771 | -7.3928 | 80.0862 | 65.7428 | 1160 | True |
| STK_055 | rank=combo_rank; threshold=balanced; concentration=diversified; exit=hold; size=equal | 4.6748 | -12.1732 | 82.1390 | 137.2822 | 935 | True |
| STK_037 | rank=win_probability; threshold=strict; concentration=diversified; exit=hold; size=equal | 4.6569 | -7.2791 | 81.4660 | 101.8084 | 955 | True |
| STK_025 | rank=ml_score; threshold=elite; concentration=risk_balanced; exit=hold; size=vol_inverse | 4.6197 | -4.7111 | 81.4885 | 63.2981 | 1048 | True |
| STK_079 | rank=combo_rank; threshold=elite; concentration=risk_balanced; exit=hold; size=vol_inverse | 4.5543 | -6.5400 | 80.3622 | 61.8769 | 1049 | True |
| STK_064 | rank=combo_rank; threshold=strict; concentration=diversified; exit=hold; size=equal | 4.5501 | -9.6922 | 81.1196 | 125.9296 | 911 | True |
| STK_035 | rank=win_probability; threshold=balanced; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 4.5470 | -9.8213 | 88.6606 | 21.1557 | 1217 | True |
| STK_028 | rank=win_probability; threshold=balanced; concentration=diversified; exit=hold; size=equal | 4.4921 | -15.1594 | 80.0821 | 90.6922 | 974 | True |
| STK_073 | rank=combo_rank; threshold=elite; concentration=diversified; exit=hold; size=equal | 4.4674 | -10.9343 | 81.8935 | 118.8326 | 845 | True |
| STK_010 | rank=ml_score; threshold=strict; concentration=diversified; exit=hold; size=equal | 4.4629 | -15.6562 | 82.3529 | 126.9186 | 918 | True |
| STK_038 | rank=win_probability; threshold=strict; concentration=diversified; exit=target4_trail2; size=equal | 4.4430 | -12.9152 | 89.1099 | 37.2552 | 955 | True |
| STK_046 | rank=win_probability; threshold=elite; concentration=diversified; exit=hold; size=equal | 4.4189 | -11.3605 | 80.4299 | 89.4067 | 884 | True |
| STK_001 | rank=ml_score; threshold=balanced; concentration=diversified; exit=hold; size=equal | 4.3914 | -13.6744 | 81.9829 | 125.8611 | 938 | True |
| STK_071 | rank=combo_rank; threshold=strict; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 4.2932 | -6.3173 | 88.9085 | 20.0402 | 1136 | True |
| STK_019 | rank=ml_score; threshold=elite; concentration=diversified; exit=hold; size=equal | 4.2789 | -11.6812 | 81.5511 | 118.9466 | 851 | True |
| STK_017 | rank=ml_score; threshold=strict; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 4.2214 | -5.9587 | 88.7225 | 19.3047 | 1135 | True |
| STK_056 | rank=combo_rank; threshold=balanced; concentration=diversified; exit=target4_trail2; size=equal | 4.1708 | -11.6453 | 89.4118 | 36.2859 | 935 | True |
| STK_065 | rank=combo_rank; threshold=strict; concentration=diversified; exit=target4_trail2; size=equal | 4.0400 | -12.7095 | 89.5719 | 34.9809 | 911 | True |
| STK_011 | rank=ml_score; threshold=strict; concentration=diversified; exit=target4_trail2; size=equal | 4.0124 | -16.1827 | 89.9782 | 35.4321 | 918 | True |
| STK_062 | rank=combo_rank; threshold=balanced; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 3.9878 | -8.8794 | 88.7069 | 19.4851 | 1160 | True |
| STK_008 | rank=ml_score; threshold=balanced; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 3.9853 | -9.5548 | 89.0878 | 20.2228 | 1173 | True |
| STK_080 | rank=combo_rank; threshold=elite; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 3.9636 | -8.8786 | 89.2278 | 18.2194 | 1049 | True |
| STK_047 | rank=win_probability; threshold=elite; concentration=diversified; exit=target4_trail2; size=equal | 3.9099 | -17.8090 | 88.4615 | 33.1498 | 884 | True |
| STK_041 | rank=win_probability; threshold=strict; concentration=focused; exit=target4_trail2; size=confidence | 3.8999 | -15.0360 | 89.2384 | 41.6678 | 604 | True |
| STK_029 | rank=win_probability; threshold=balanced; concentration=diversified; exit=target4_trail2; size=equal | 3.8885 | -20.5608 | 88.0903 | 34.6232 | 974 | True |
| STK_040 | rank=win_probability; threshold=strict; concentration=focused; exit=hold; size=confidence | 3.8638 | -11.3666 | 80.7947 | 102.5861 | 604 | True |
| STK_074 | rank=combo_rank; threshold=elite; concentration=diversified; exit=target4_trail2; size=equal | 3.8561 | -12.2400 | 89.1124 | 31.3570 | 845 | True |
| STK_053 | rank=win_probability; threshold=elite; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 3.8428 | -9.6655 | 88.4372 | 18.1461 | 1107 | True |
| STK_013 | rank=ml_score; threshold=strict; concentration=focused; exit=hold; size=confidence | 3.8296 | -15.7206 | 83.1919 | 144.3538 | 589 | True |
| STK_002 | rank=ml_score; threshold=balanced; concentration=diversified; exit=target4_trail2; size=equal | 3.7904 | -13.5049 | 89.6588 | 33.9767 | 938 | True |
| STK_067 | rank=combo_rank; threshold=strict; concentration=focused; exit=hold; size=confidence | 3.7508 | -13.8945 | 82.6011 | 138.3130 | 569 | True |
| STK_031 | rank=win_probability; threshold=balanced; concentration=focused; exit=hold; size=confidence | 3.7466 | -13.0645 | 79.9674 | 97.8730 | 614 | True |
| STK_026 | rank=ml_score; threshold=elite; concentration=risk_balanced; exit=target4_trail2; size=vol_inverse | 3.7296 | -7.8445 | 88.9313 | 17.7701 | 1048 | True |
| STK_022 | rank=ml_score; threshold=elite; concentration=focused; exit=hold; size=confidence | 3.7190 | -13.5927 | 82.5279 | 121.1158 | 538 | True |
| STK_004 | rank=ml_score; threshold=balanced; concentration=focused; exit=hold; size=confidence | 3.6683 | -17.1789 | 82.7586 | 140.3138 | 580 | True |
| STK_020 | rank=ml_score; threshold=elite; concentration=diversified; exit=target4_trail2; size=equal | 3.6627 | -13.5289 | 89.6592 | 31.5404 | 851 | True |
| STK_058 | rank=combo_rank; threshold=balanced; concentration=focused; exit=hold; size=confidence | 3.6210 | -16.5177 | 82.9352 | 142.2715 | 586 | True |
| STK_032 | rank=win_probability; threshold=balanced; concentration=focused; exit=target4_trail2; size=confidence | 3.6052 | -15.8682 | 87.7850 | 40.1344 | 614 | True |
| STK_049 | rank=win_probability; threshold=elite; concentration=focused; exit=hold; size=confidence | 3.5569 | -12.1716 | 80.0718 | 81.1070 | 557 | True |
| STK_076 | rank=combo_rank; threshold=elite; concentration=focused; exit=hold; size=confidence | 3.3570 | -16.5501 | 81.4471 | 114.8079 | 539 | True |
| STK_050 | rank=win_probability; threshold=elite; concentration=focused; exit=target4_trail2; size=confidence | 3.3469 | -17.0701 | 89.4075 | 33.3926 | 557 | True |
| STK_023 | rank=ml_score; threshold=elite; concentration=focused; exit=target4_trail2; size=confidence | 3.2960 | -15.4146 | 89.9628 | 31.6006 | 538 | True |
| STK_059 | rank=combo_rank; threshold=balanced; concentration=focused; exit=target4_trail2; size=confidence | 3.2078 | -19.8319 | 89.4198 | 36.7783 | 586 | True |
| STK_068 | rank=combo_rank; threshold=strict; concentration=focused; exit=target4_trail2; size=confidence | 3.1382 | -13.8426 | 89.2794 | 35.5345 | 569 | True |
| STK_005 | rank=ml_score; threshold=balanced; concentration=focused; exit=target4_trail2; size=confidence | 3.0601 | -15.7349 | 89.1379 | 34.4092 | 580 | True |
| STK_077 | rank=combo_rank; threshold=elite; concentration=focused; exit=target4_trail2; size=confidence | 2.9500 | -16.0094 | 89.9814 | 30.3911 | 539 | True |
| STK_014 | rank=ml_score; threshold=strict; concentration=focused; exit=target4_trail2; size=confidence | 2.7388 | -22.4016 | 89.4737 | 33.8999 | 589 | False |

### Pareto Frontier
| strategy_id | sharpe | max_drawdown | annualized_return | trade_count |
| --- | --- | --- | --- | --- |
| STK_045 | 7.1102 | -2.6529 | 35.6734 | 1194 |
| STK_054 | 6.8839 | -2.0164 | 33.6428 | 1107 |

### Top 10 Detailed
#### STK_045
- Config: rank=win_probability; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 7.11
- MaxDD: -2.65%
- CAGR: 35.67%
- Win rate: 84.51%
- Trades: 1194
- Yearly returns: 2018: 32.64%, 2019: 26.31%, 2020: 31.22%, 2021: 42.36%, 2022: 40.18%, 2023: 41.42%, 2024: 31.33%, 2025: 38.25%

#### STK_063
- Config: rank=combo_rank; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 7.06
- MaxDD: -3.22%
- CAGR: 35.23%
- Win rate: 83.71%
- Trades: 1160
- Yearly returns: 2018: 26.41%, 2019: 30.82%, 2020: 26.16%, 2021: 38.90%, 2022: 44.46%, 2023: 38.72%, 2024: 35.32%, 2025: 39.65%

#### STK_072
- Config: rank=combo_rank; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 7.05
- MaxDD: -3.22%
- CAGR: 34.95%
- Win rate: 84.33%
- Trades: 1136
- Yearly returns: 2018: 26.52%, 2019: 25.39%, 2020: 24.84%, 2021: 39.44%, 2022: 44.63%, 2023: 38.46%, 2024: 38.62%, 2025: 39.65%

#### STK_036
- Config: rank=win_probability; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 7.00
- MaxDD: -2.96%
- CAGR: 36.26%
- Win rate: 84.06%
- Trades: 1217
- Yearly returns: 2018: 33.77%, 2019: 33.73%, 2020: 29.19%, 2021: 38.75%, 2022: 43.21%, 2023: 41.19%, 2024: 31.33%, 2025: 38.25%

#### STK_009
- Config: rank=ml_score; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 6.94
- MaxDD: -2.67%
- CAGR: 36.23%
- Win rate: 84.40%
- Trades: 1173
- Yearly returns: 2018: 27.94%, 2019: 30.06%, 2020: 25.95%, 2021: 39.13%, 2022: 41.71%, 2023: 41.70%, 2024: 37.88%, 2025: 42.75%

#### STK_018
- Config: rank=ml_score; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 6.94
- MaxDD: -3.08%
- CAGR: 34.64%
- Win rate: 84.67%
- Trades: 1135
- Yearly returns: 2018: 25.53%, 2019: 26.24%, 2020: 25.04%, 2021: 37.90%, 2022: 39.35%, 2023: 42.07%, 2024: 39.53%, 2025: 40.17%

#### STK_054
- Config: rank=win_probability; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 6.88
- MaxDD: -2.02%
- CAGR: 33.64%
- Win rate: 84.64%
- Trades: 1107
- Yearly returns: 2018: 31.20%, 2019: 8.33%, 2020: 28.41%, 2021: 37.99%, 2022: 44.72%, 2023: 45.04%, 2024: 32.30%, 2025: 41.76%

#### STK_027
- Config: rank=ml_score; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 6.72
- MaxDD: -2.07%
- CAGR: 32.57%
- Win rate: 84.73%
- Trades: 1048
- Yearly returns: 2018: 28.30%, 2019: 10.56%, 2020: 23.45%, 2021: 35.97%, 2022: 42.65%, 2023: 43.30%, 2024: 35.02%, 2025: 41.09%

#### STK_057
- Config: rank=combo_rank; threshold=balanced; concentration=diversified; exit=target6_atr2; size=equal
- Sharpe: 6.68
- MaxDD: -6.01%
- CAGR: 64.99%
- Win rate: 85.35%
- Trades: 935
- Yearly returns: 2018: 49.77%, 2019: 55.50%, 2020: 46.99%, 2021: 72.78%, 2022: 80.62%, 2023: 76.17%, 2024: 64.01%, 2025: 73.31%

#### STK_081
- Config: rank=combo_rank; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse
- Sharpe: 6.60
- MaxDD: -3.22%
- CAGR: 31.83%
- Win rate: 83.98%
- Trades: 1049
- Yearly returns: 2018: 29.60%, 2019: 9.54%, 2020: 21.78%, 2021: 37.06%, 2022: 44.58%, 2023: 37.11%, 2024: 36.43%, 2025: 39.90%


### Holdout Check (Top 5 on 2018-2024 vs 2025)
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| STK_045 | 2018_2024 | 6.9474 | -2.6529 | 34.7638 | 84.0734 | 1036 |
| STK_045 | 2025 | 7.1535 | -0.9318 | 38.8286 | 85.8025 | 162 |
| STK_063 | 2018_2024 | 6.9006 | -3.2162 | 34.0136 | 82.9724 | 1016 |
| STK_063 | 2025 | 8.3219 | -0.8204 | 43.0707 | 89.7959 | 147 |
| STK_072 | 2018_2024 | 6.8874 | -3.2162 | 33.6957 | 83.6694 | 992 |
| STK_072 | 2025 | 8.3219 | -0.8204 | 43.0707 | 89.7959 | 147 |
| STK_036 | 2018_2024 | 6.8235 | -2.9649 | 35.4341 | 83.5694 | 1059 |
| STK_036 | 2025 | 7.1535 | -0.9318 | 38.8286 | 85.8025 | 162 |
| STK_009 | 2018_2024 | 6.8155 | -2.6663 | 35.0077 | 83.4951 | 1030 |
| STK_009 | 2025 | 8.0000 | -0.8662 | 43.5347 | 91.9463 | 149 |

### Key Findings
- Best combined stock strategy: `STK_045` with Sharpe 7.11, max drawdown -2.65%, and annualized return 35.67%.
- Average Sharpe by ranking signal was led by `win_probability` at 4.88.
- Average Sharpe by exit profile was led by `target6_atr2` at 6.20.


### Long-Only Stock Results
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STK_063 | rank=combo_rank; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.8632 | -3.2162 | 83.8462 | 34.7001 | 1170 | True |
| STK_009 | rank=ml_score; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.7890 | -3.5899 | 83.7998 | 34.4071 | 1179 | True |
| STK_072 | rank=combo_rank; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.7627 | -3.2162 | 83.3775 | 32.5263 | 1131 | True |
| STK_018 | rank=ml_score; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.7144 | -3.0756 | 83.4211 | 32.6700 | 1140 | True |
| STK_045 | rank=win_probability; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.7072 | -3.2776 | 83.5314 | 33.3805 | 1178 | True |
| STK_081 | rank=combo_rank; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.6090 | -3.2162 | 84.3564 | 30.3424 | 1010 | True |
| STK_036 | rank=win_probability; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.5381 | -3.2469 | 83.3059 | 34.3052 | 1216 | True |
| STK_027 | rank=ml_score; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.4969 | -2.0698 | 84.0353 | 30.1571 | 1021 | True |
| STK_054 | rank=win_probability; threshold=elite; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 6.3646 | -2.6442 | 83.8619 | 30.1638 | 1072 | True |
| STK_039 | rank=win_probability; threshold=strict; concentration=diversified; exit=target6_atr2; size=equal | 6.2913 | -4.4294 | 83.9958 | 58.6483 | 956 | True |

#### Holdout Check
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| STK_063 | 2018_2024 | 6.7273 | -3.2162 | 33.7110 | 83.2359 | 1026 |
| STK_063 | 2025 | 9.2149 | -0.6287 | 43.7972 | 89.9329 | 149 |
| STK_009 | 2018_2024 | 6.6390 | -3.5899 | 33.4028 | 83.3656 | 1034 |
| STK_009 | 2025 | 7.9596 | -0.7929 | 39.0242 | 87.7551 | 147 |
| STK_072 | 2018_2024 | 6.5894 | -3.2162 | 31.3771 | 82.8109 | 989 |
| STK_072 | 2025 | 9.2149 | -0.6287 | 43.7972 | 89.9329 | 149 |
| STK_018 | 2018_2024 | 6.4329 | -3.0756 | 31.1252 | 82.3647 | 998 |
| STK_018 | 2025 | 7.9596 | -0.7929 | 39.0242 | 87.7551 | 147 |
| STK_045 | 2018_2024 | 6.5236 | -3.2776 | 31.9776 | 82.7451 | 1020 |
| STK_045 | 2025 | 7.7062 | -1.0032 | 40.5451 | 87.5776 | 161 |

- Best long-only stock results strategy: `STK_063` with Sharpe 6.86, max drawdown -3.22%, and annualized return 34.70%.
- Average Sharpe by ranking signal was led by `ml_score` at 4.35.
- Average Sharpe by exit profile was led by `target6_atr2` at 5.91.


### Short-Only Stock Results
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STK_009 | rank=ml_score; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 5.5122 | -2.1308 | 86.1650 | 22.6108 | 824 | True |
| STK_018 | rank=ml_score; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 5.4177 | -1.5473 | 86.1496 | 20.6988 | 722 | False |
| STK_072 | rank=combo_rank; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 5.2981 | -2.1040 | 86.6109 | 20.5474 | 717 | False |
| STK_063 | rank=combo_rank; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 5.2248 | -2.5520 | 84.8965 | 22.3823 | 821 | True |
| STK_003 | rank=ml_score; threshold=balanced; concentration=diversified; exit=target6_atr2; size=equal | 5.2172 | -3.5500 | 86.1194 | 39.8970 | 670 | True |
| STK_012 | rank=ml_score; threshold=strict; concentration=diversified; exit=target6_atr2; size=equal | 5.2089 | -3.2650 | 88.2051 | 36.9592 | 585 | False |
| STK_036 | rank=win_probability; threshold=balanced; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 5.2039 | -1.9755 | 83.6639 | 21.2344 | 857 | True |
| STK_066 | rank=combo_rank; threshold=strict; concentration=diversified; exit=target6_atr2; size=equal | 5.0880 | -3.0352 | 87.6712 | 35.6399 | 584 | False |
| STK_057 | rank=combo_rank; threshold=balanced; concentration=diversified; exit=target6_atr2; size=equal | 5.0782 | -3.1936 | 85.4573 | 37.7617 | 667 | True |
| STK_045 | rank=win_probability; threshold=strict; concentration=risk_balanced; exit=target6_atr2; size=vol_inverse | 5.0651 | -3.5632 | 84.6774 | 19.4188 | 744 | False |

#### Holdout Check
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| STK_009 | 2018_2024 | 5.0133 | -2.1308 | 19.8668 | 85.1190 | 672 |
| STK_009 | 2025 | 8.2686 | -1.1381 | 40.6325 | 90.3226 | 155 |
| STK_018 | 2018_2024 | 5.0496 | -1.5473 | 18.8121 | 85.3492 | 587 |
| STK_018 | 2025 | 7.5623 | -0.7964 | 32.5521 | 89.7059 | 136 |
| STK_072 | 2018_2024 | 4.9231 | -2.1040 | 18.5166 | 85.8621 | 580 |
| STK_072 | 2025 | 7.4510 | -0.7964 | 33.1756 | 89.7810 | 137 |
| STK_063 | 2018_2024 | 4.6938 | -2.5520 | 19.3792 | 83.5082 | 667 |
| STK_063 | 2025 | 8.8410 | -0.5649 | 43.4810 | 91.0828 | 157 |
| STK_003 | 2018_2024 | 4.7877 | -3.5500 | 35.3150 | 85.0554 | 542 |
| STK_003 | 2025 | 7.7481 | -2.7024 | 66.3703 | 89.8438 | 128 |

- Best short-only stock results strategy: `STK_009` with Sharpe 5.51, max drawdown -2.13%, and annualized return 22.61%.
- Average Sharpe by ranking signal was led by `ml_score` at 4.06.
- Average Sharpe by exit profile was led by `target6_atr2` at 4.70.


## Options Strategy Results
### Combined Long + Short
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPT_013 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.10; size=vol_inverse | 5.3438 | -22.3235 | 71.1790 | 3958.9647 | 687 | True |
| OPT_014 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.18; size=vol_inverse | 5.1940 | -24.0915 | 69.7234 | 3524.1537 | 687 | True |
| OPT_015 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.25; size=vol_inverse | 5.0630 | -25.4067 | 68.1223 | 3181.9005 | 687 | True |
| OPT_031 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.025_theta_0.10; size=vol_inverse | 5.0533 | -24.0602 | 72.8499 | 3244.5527 | 593 | True |
| OPT_007 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.10; size=equal | 4.9601 | -30.2646 | 71.5532 | 33007.3354 | 573 | True |
| OPT_032 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.025_theta_0.18; size=vol_inverse | 4.9338 | -25.6355 | 72.3440 | 2948.4391 | 593 | True |
| OPT_033 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.025_theta_0.25; size=vol_inverse | 4.8297 | -26.8880 | 71.5008 | 2712.6388 | 593 | True |
| OPT_008 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.18; size=equal | 4.8255 | -32.4248 | 69.8080 | 27905.7370 | 573 | True |
| OPT_016 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.030_theta_0.10; size=vol_inverse | 4.7855 | -25.7932 | 66.0844 | 2196.9899 | 687 | True |
| OPT_009 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.25; size=equal | 4.7086 | -33.9202 | 69.2845 | 24088.1616 | 573 | True |
| OPT_034 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.030_theta_0.10; size=vol_inverse | 4.6047 | -27.2786 | 70.1518 | 1976.0990 | 593 | True |
| OPT_017 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.030_theta_0.18; size=vol_inverse | 4.6032 | -27.5269 | 64.9199 | 1915.3264 | 687 | True |
| OPT_025 | rank=combo_rank; threshold=elite; concentration=balanced; exit=premium_0.025_theta_0.10; size=equal | 4.5635 | -38.1542 | 72.0322 | 17520.3262 | 497 | True |
| OPT_010 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.030_theta_0.10; size=equal | 4.4614 | -33.4482 | 67.5393 | 13893.6338 | 573 | True |
| OPT_035 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.030_theta_0.18; size=vol_inverse | 4.4598 | -28.7969 | 68.9713 | 1764.4111 | 593 | True |
| OPT_018 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.030_theta_0.25; size=vol_inverse | 4.4429 | -28.9825 | 63.6099 | 1698.3229 | 687 | True |
| OPT_026 | rank=combo_rank; threshold=elite; concentration=balanced; exit=premium_0.025_theta_0.18; size=equal | 4.4406 | -40.3463 | 70.8249 | 15094.3466 | 497 | True |
| OPT_027 | rank=combo_rank; threshold=elite; concentration=balanced; exit=premium_0.025_theta_0.25; size=equal | 4.3370 | -41.5132 | 69.8189 | 13278.1786 | 497 | True |
| OPT_036 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.030_theta_0.25; size=vol_inverse | 4.3351 | -30.0596 | 68.2968 | 1599.7224 | 593 | True |
| OPT_011 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.030_theta_0.18; size=equal | 4.2961 | -35.4539 | 66.4921 | 11383.8613 | 573 | True |
| OPT_001 | rank=combo_rank; threshold=strict; concentration=compact; exit=premium_0.025_theta_0.10; size=confidence | 4.2830 | -49.7943 | 71.8615 | 34000.3729 | 462 | True |
| OPT_019 | rank=combo_rank; threshold=elite; concentration=compact; exit=premium_0.025_theta_0.10; size=confidence | 4.2510 | -36.1560 | 74.0831 | 21069.5506 | 409 | True |
| OPT_002 | rank=combo_rank; threshold=strict; concentration=compact; exit=premium_0.025_theta_0.18; size=confidence | 4.1608 | -52.2160 | 70.7792 | 28300.1169 | 462 | True |
| OPT_012 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.030_theta_0.25; size=equal | 4.1515 | -37.0264 | 65.6195 | 9561.7684 | 573 | True |
| OPT_020 | rank=combo_rank; threshold=elite; concentration=compact; exit=premium_0.025_theta_0.18; size=confidence | 4.1454 | -37.3794 | 73.8386 | 18208.9477 | 409 | True |
| OPT_028 | rank=combo_rank; threshold=elite; concentration=balanced; exit=premium_0.030_theta_0.10; size=equal | 4.1279 | -42.7434 | 68.4105 | 8655.4329 | 497 | True |
| OPT_003 | rank=combo_rank; threshold=strict; concentration=compact; exit=premium_0.025_theta_0.25; size=confidence | 4.0540 | -53.8232 | 69.6970 | 24086.7060 | 462 | True |
| OPT_021 | rank=combo_rank; threshold=elite; concentration=compact; exit=premium_0.025_theta_0.25; size=confidence | 4.0535 | -41.3189 | 72.3716 | 16024.1790 | 409 | True |
| OPT_029 | rank=combo_rank; threshold=elite; concentration=balanced; exit=premium_0.030_theta_0.18; size=equal | 3.9812 | -45.1634 | 67.6056 | 7289.8732 | 497 | True |
| OPT_022 | rank=combo_rank; threshold=elite; concentration=compact; exit=premium_0.030_theta_0.10; size=confidence | 3.8608 | -43.9044 | 70.1711 | 10305.3215 | 409 | True |
| OPT_030 | rank=combo_rank; threshold=elite; concentration=balanced; exit=premium_0.030_theta_0.25; size=equal | 3.8559 | -47.1633 | 66.5996 | 6283.1051 | 497 | True |
| OPT_004 | rank=combo_rank; threshold=strict; concentration=compact; exit=premium_0.030_theta_0.10; size=confidence | 3.8318 | -53.1840 | 67.3160 | 13934.6724 | 462 | True |
| OPT_023 | rank=combo_rank; threshold=elite; concentration=compact; exit=premium_0.030_theta_0.18; size=confidence | 3.7333 | -48.4492 | 69.1932 | 8693.2068 | 409 | True |
| OPT_005 | rank=combo_rank; threshold=strict; concentration=compact; exit=premium_0.030_theta_0.18; size=confidence | 3.6814 | -55.3636 | 66.4502 | 11243.4975 | 462 | True |
| OPT_024 | rank=combo_rank; threshold=elite; concentration=compact; exit=premium_0.030_theta_0.25; size=confidence | 3.6207 | -52.3505 | 67.9707 | 7485.4685 | 409 | True |
| OPT_006 | rank=combo_rank; threshold=strict; concentration=compact; exit=premium_0.030_theta_0.25; size=confidence | 3.5492 | -57.0236 | 65.1515 | 9311.0149 | 462 | True |

### Pareto Frontier
| strategy_id | sharpe | max_drawdown | annualized_return | trade_count |
| --- | --- | --- | --- | --- |
| OPT_013 | 5.3438 | -22.3235 | 3958.9647 | 687 |

### Holdout Check (Top 5 on 2018-2024 vs 2025)
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| OPT_013 | 2018_2024 | 5.2139 | -22.3235 | 3213.2285 | 69.7171 | 601 |
| OPT_013 | 2025 | 5.9642 | -8.8384 | 10564.8905 | 82.2222 | 90 |
| OPT_014 | 2018_2024 | 5.0555 | -24.0915 | 2843.7884 | 68.2196 | 601 |
| OPT_014 | 2025 | 5.8528 | -9.8890 | 9603.7079 | 77.7778 | 90 |
| OPT_015 | 2018_2024 | 4.9174 | -25.4067 | 2554.5725 | 66.5557 | 601 |
| OPT_015 | 2025 | 5.7546 | -10.6348 | 8840.5310 | 77.7778 | 90 |
| OPT_031 | 2018_2024 | 5.1360 | -24.0602 | 2856.0239 | 72.8880 | 509 |
| OPT_031 | 2025 | 5.1811 | -8.8384 | 6137.6078 | 74.7126 | 87 |
| OPT_007 | 2018_2024 | 4.8468 | -30.2646 | 28571.3800 | 70.8583 | 501 |
| OPT_007 | 2025 | 5.2609 | -16.1580 | 91648.9338 | 80.0000 | 75 |

### Key Findings
- Best combined options strategy: `OPT_013` with Sharpe 5.34, max drawdown -22.32%, and annualized return 3958.96%.
- Average Sharpe by ranking signal was led by `combo_rank` at 4.38.
- Average Sharpe by exit profile was led by `premium_0.025_theta_0.10` at 4.74.


### Long-Only Options Results
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPT_013 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.10; size=vol_inverse | 4.9697 | -30.4193 | 68.5507 | 2938.1911 | 690 | True |
| OPT_031 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.025_theta_0.10; size=vol_inverse | 4.8744 | -24.9775 | 69.3493 | 2258.3116 | 584 | True |
| OPT_014 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.18; size=vol_inverse | 4.8034 | -32.0059 | 66.6667 | 2591.9000 | 690 | True |
| OPT_032 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.025_theta_0.18; size=vol_inverse | 4.7336 | -26.0356 | 68.3219 | 2037.5740 | 584 | True |
| OPT_015 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.25; size=vol_inverse | 4.6616 | -33.3167 | 65.2174 | 2325.5423 | 690 | True |
| OPT_007 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.10; size=equal | 4.6575 | -48.2100 | 68.2759 | 20511.3991 | 580 | True |
| OPT_033 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.025_theta_0.25; size=vol_inverse | 4.6121 | -26.9595 | 67.4658 | 1863.4849 | 584 | True |
| OPT_008 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.18; size=equal | 4.5088 | -51.4356 | 66.2069 | 17098.4731 | 580 | True |
| OPT_009 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.25; size=equal | 4.3828 | -53.9644 | 65.8621 | 14619.6771 | 580 | True |
| OPT_034 | rank=combo_rank; threshold=elite; concentration=wide; exit=premium_0.030_theta_0.10; size=vol_inverse | 4.3788 | -27.0456 | 65.5822 | 1363.8051 | 584 | True |

#### Holdout Check
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| OPT_013 | 2018_2024 | 4.8463 | -30.4193 | 2569.3021 | 67.2757 | 602 |
| OPT_013 | 2025 | 5.5055 | -9.8973 | 5063.3274 | 75.5556 | 90 |
| OPT_031 | 2018_2024 | 4.7663 | -24.9775 | 2022.4599 | 69.1383 | 499 |
| OPT_031 | 2025 | 5.3227 | -9.8973 | 3935.4132 | 69.7674 | 86 |
| OPT_014 | 2018_2024 | 4.6764 | -32.0059 | 2262.0506 | 65.2824 | 602 |
| OPT_014 | 2025 | 5.3529 | -10.7193 | 4511.3700 | 72.2222 | 90 |
| OPT_032 | 2018_2024 | 4.6317 | -26.0356 | 1827.3442 | 68.1363 | 499 |
| OPT_032 | 2025 | 5.1455 | -10.7193 | 3515.1631 | 66.2791 | 86 |
| OPT_015 | 2018_2024 | 4.5317 | -33.3167 | 2025.8393 | 63.7874 | 602 |
| OPT_015 | 2025 | 5.2213 | -11.2649 | 4084.2284 | 71.1111 | 90 |

- Best long-only options results strategy: `OPT_013` with Sharpe 4.97, max drawdown -30.42%, and annualized return 2938.19%.
- Average Sharpe by ranking signal was led by `combo_rank` at 4.07.
- Average Sharpe by exit profile was led by `premium_0.025_theta_0.10` at 4.47.


### Short-Only Options Results
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPT_013 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.10; size=vol_inverse | 3.9661 | -15.3860 | 69.9541 | 743.4598 | 436 | False |
| OPT_014 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.18; size=vol_inverse | 3.8247 | -16.7676 | 68.5780 | 671.7099 | 436 | False |
| OPT_015 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.025_theta_0.25; size=vol_inverse | 3.6975 | -17.8422 | 68.3486 | 613.2467 | 436 | False |
| OPT_007 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.10; size=equal | 3.4203 | -28.0668 | 68.3924 | 2239.6809 | 367 | False |
| OPT_016 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.030_theta_0.10; size=vol_inverse | 3.3807 | -18.7373 | 66.0550 | 433.1626 | 436 | False |
| OPT_008 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.18; size=equal | 3.2922 | -29.3855 | 67.3025 | 1937.1208 | 367 | False |
| OPT_017 | rank=combo_rank; threshold=strict; concentration=wide; exit=premium_0.030_theta_0.18; size=vol_inverse | 3.2002 | -20.6370 | 64.9083 | 381.1402 | 436 | False |
| OPT_001 | rank=combo_rank; threshold=strict; concentration=compact; exit=premium_0.025_theta_0.10; size=confidence | 3.1863 | -33.6421 | 68.1063 | 2344.0761 | 301 | False |
| OPT_009 | rank=combo_rank; threshold=strict; concentration=balanced; exit=premium_0.025_theta_0.25; size=equal | 3.1763 | -30.5695 | 66.7575 | 1699.8949 | 367 | False |
| OPT_002 | rank=combo_rank; threshold=strict; concentration=compact; exit=premium_0.025_theta_0.18; size=confidence | 3.0645 | -37.7919 | 66.4452 | 2014.0258 | 301 | False |

#### Holdout Check
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| OPT_013 | 2018_2024 | 3.8017 | -15.3860 | 633.2043 | 70.2857 | 350 |
| OPT_013 | 2025 | 5.1549 | -10.3652 | 2073.7491 | 70.9302 | 86 |
| OPT_014 | 2018_2024 | 3.6686 | -16.7676 | 575.5703 | 68.5714 | 350 |
| OPT_014 | 2025 | 4.9675 | -11.5400 | 1802.0373 | 70.9302 | 86 |
| OPT_015 | 2018_2024 | 3.5493 | -17.8422 | 528.4885 | 68.2857 | 350 |
| OPT_015 | 2025 | 4.7947 | -12.8034 | 1586.0425 | 70.9302 | 86 |
| OPT_007 | 2018_2024 | 3.2700 | -28.0668 | 1764.7097 | 68.5811 | 296 |
| OPT_007 | 2025 | 4.5270 | -16.7123 | 12243.8450 | 71.2329 | 73 |
| OPT_016 | 2018_2024 | 3.2670 | -18.7373 | 383.4644 | 66.5714 | 350 |
| OPT_016 | 2025 | 4.2710 | -14.3628 | 904.7043 | 66.2791 | 86 |

- Best short-only options results strategy: `OPT_013` with Sharpe 3.97, max drawdown -15.39%, and annualized return 743.46%.
- Average Sharpe by ranking signal was led by `combo_rank` at 2.74.
- Average Sharpe by exit profile was led by `premium_0.025_theta_0.10` at 3.07.


## Spread Strategy Results
### Combined Long + Short
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPR_048 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_2.0_8.0; size=vol_inverse | 6.5159 | -25.0733 | 90.1124 | 613.4709 | 890 | True |
| SPR_036 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_2.0_8.0; size=vol_inverse | 6.3920 | -25.0733 | 89.2741 | 640.8448 | 923 | True |
| SPR_042 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_1.0_8.0; size=vol_inverse | 6.0605 | -27.9460 | 88.5393 | 542.2358 | 890 | True |
| SPR_046 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_2.0_6.0; size=vol_inverse | 6.0287 | -29.7696 | 89.3258 | 553.2783 | 890 | True |
| SPR_030 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_1.0_8.0; size=vol_inverse | 5.8789 | -27.9460 | 87.7573 | 556.6465 | 923 | True |
| SPR_034 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_2.0_6.0; size=vol_inverse | 5.8516 | -29.7696 | 88.5157 | 568.9503 | 923 | True |
| SPR_047 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_put_2.0_8.0; size=equal | 5.6134 | -43.5669 | 88.7073 | 1374.8314 | 673 | True |
| SPR_035 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_2.0_8.0; size=equal | 5.5968 | -43.5669 | 89.3314 | 1649.9420 | 703 | True |
| SPR_040 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_1.0_6.0; size=vol_inverse | 5.5519 | -31.3216 | 87.5281 | 480.0984 | 890 | True |
| SPR_044 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_2.0_4.0; size=vol_inverse | 5.5016 | -32.3763 | 88.2022 | 487.0397 | 890 | True |
| SPR_028 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_1.0_6.0; size=vol_inverse | 5.2853 | -30.4012 | 86.4572 | 482.7079 | 923 | True |
| SPR_032 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_2.0_4.0; size=vol_inverse | 5.2189 | -30.4011 | 87.3239 | 489.5395 | 923 | True |
| SPR_029 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_1.0_8.0; size=equal | 5.1447 | -49.1906 | 88.3357 | 1384.0322 | 703 | True |
| SPR_041 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_put_1.0_8.0; size=equal | 5.0922 | -49.1906 | 87.5186 | 1129.3501 | 673 | True |
| SPR_033 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_2.0_6.0; size=equal | 5.0765 | -52.8012 | 88.9047 | 1402.4750 | 703 | True |
| SPR_045 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_put_2.0_6.0; size=equal | 5.0198 | -52.8012 | 88.2615 | 1146.3649 | 673 | True |
| SPR_038 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_1.0_4.0; size=vol_inverse | 4.9947 | -33.0363 | 86.2921 | 413.4952 | 890 | True |
| SPR_027 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_1.0_6.0; size=equal | 4.6572 | -54.0396 | 87.1977 | 1164.3508 | 703 | True |
| SPR_026 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_1.0_4.0; size=vol_inverse | 4.6470 | -32.8023 | 85.1571 | 405.3707 | 923 | True |
| SPR_031 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_2.0_4.0; size=equal | 4.6311 | -54.0396 | 87.9090 | 1187.2120 | 703 | True |
| SPR_039 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_put_1.0_6.0; size=equal | 4.5189 | -54.0396 | 86.3299 | 925.7175 | 673 | True |
| SPR_014 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_call_1.0_4.0; size=vol_inverse | 4.5038 | -36.2039 | 84.3820 | 388.5047 | 890 | True |
| SPR_043 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_put_2.0_4.0; size=equal | 4.4451 | -56.6549 | 87.2214 | 935.2704 | 673 | True |
| SPR_002 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_call_1.0_4.0; size=vol_inverse | 4.3389 | -44.8713 | 83.9653 | 401.0553 | 923 | True |
| SPR_025 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_1.0_4.0; size=equal | 4.1685 | -54.0396 | 86.2020 | 949.2998 | 703 | True |
| SPR_037 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_put_1.0_4.0; size=equal | 3.9064 | -61.9838 | 84.6954 | 722.2874 | 673 | True |
| SPR_020 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_call_2.0_4.0; size=vol_inverse | 3.8500 | -36.6500 | 81.1236 | 306.6163 | 890 | True |
| SPR_013 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_call_1.0_4.0; size=equal | 3.8201 | -55.1358 | 84.1010 | 787.1775 | 673 | True |
| SPR_008 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_call_2.0_4.0; size=vol_inverse | 3.6994 | -48.4898 | 81.0401 | 312.0121 | 923 | True |
| SPR_001 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_call_1.0_4.0; size=equal | 3.6469 | -75.7774 | 83.5949 | 804.5956 | 701 | True |
| SPR_016 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_call_1.0_6.0; size=vol_inverse | 3.4182 | -38.7019 | 79.8876 | 244.3456 | 890 | True |
| SPR_004 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_call_1.0_6.0; size=vol_inverse | 3.3202 | -52.4077 | 79.3066 | 250.3484 | 923 | True |
| SPR_007 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_call_2.0_4.0; size=equal | 3.1300 | -78.6089 | 80.7418 | 568.8798 | 701 | True |
| SPR_019 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_call_2.0_4.0; size=equal | 3.1158 | -65.4721 | 80.9807 | 534.6979 | 673 | True |
| SPR_022 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_call_2.0_6.0; size=vol_inverse | 3.0209 | -46.2794 | 77.8652 | 199.4978 | 890 | True |
| SPR_010 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_call_2.0_6.0; size=vol_inverse | 3.0108 | -55.1469 | 77.6815 | 211.6016 | 923 | True |
| SPR_015 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_call_1.0_6.0; size=equal | 2.8739 | -65.4776 | 80.5349 | 443.8973 | 673 | True |
| SPR_003 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_call_1.0_6.0; size=equal | 2.8531 | -80.7792 | 79.6006 | 460.5080 | 701 | False |
| SPR_018 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_call_1.0_8.0; size=vol_inverse | 2.6292 | -47.2642 | 76.1798 | 161.2918 | 890 | True |
| SPR_006 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_call_1.0_8.0; size=vol_inverse | 2.6252 | -58.6400 | 75.7313 | 169.4042 | 923 | True |
| SPR_021 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_call_2.0_6.0; size=equal | 2.5247 | -65.4778 | 78.1575 | 334.0074 | 673 | True |
| SPR_009 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_call_2.0_6.0; size=equal | 2.4401 | -89.2896 | 77.5714 | 340.1684 | 700 | False |
| SPR_017 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_call_1.0_8.0; size=equal | 2.2545 | -65.4804 | 76.5230 | 266.2170 | 673 | True |
| SPR_005 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_call_1.0_8.0; size=equal | 2.1496 | -90.5943 | 76.1429 | 262.5055 | 700 | False |
| SPR_024 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_call_2.0_8.0; size=vol_inverse | 1.9672 | -48.8789 | 73.1461 | 104.5365 | 890 | False |
| SPR_012 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_call_2.0_8.0; size=vol_inverse | 1.9313 | -61.0876 | 72.5894 | 107.4206 | 923 | True |
| SPR_023 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_call_2.0_8.0; size=equal | 1.7679 | -67.1747 | 73.8484 | 169.2225 | 673 | False |
| SPR_011 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_call_2.0_8.0; size=equal | 1.6800 | -90.7040 | 73.8571 | 163.4138 | 700 | False |

### Pareto Frontier
| strategy_id | sharpe | max_drawdown | annualized_return | trade_count |
| --- | --- | --- | --- | --- |
| SPR_048 | 6.5159 | -25.0733 | 613.4709 | 890 |

### Holdout Check (Top 5 on 2018-2024 vs 2025)
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| SPR_048 | 2018_2024 | 6.3356 | -25.0733 | 573.0443 | 89.5484 | 775 |
| SPR_048 | 2025 | 7.6716 | -5.2746 | 810.8027 | 93.2773 | 119 |
| SPR_036 | 2018_2024 | 6.2147 | -25.0733 | 605.2827 | 88.8337 | 806 |
| SPR_036 | 2025 | 7.6716 | -5.2746 | 810.8027 | 93.2773 | 119 |
| SPR_042 | 2018_2024 | 5.8495 | -27.9460 | 501.9840 | 87.7419 | 775 |
| SPR_042 | 2025 | 7.3300 | -5.7250 | 739.2651 | 92.4370 | 119 |
| SPR_046 | 2018_2024 | 5.8190 | -29.7696 | 511.7442 | 88.6452 | 775 |
| SPR_046 | 2025 | 7.2622 | -6.9874 | 756.0256 | 92.4370 | 119 |
| SPR_030 | 2018_2024 | 5.6870 | -27.9460 | 522.4171 | 87.0968 | 806 |
| SPR_030 | 2025 | 7.3300 | -5.7250 | 739.2651 | 92.4370 | 119 |

### Key Findings
- Best combined spread strategy: `SPR_048` with Sharpe 6.52, max drawdown -25.07%, and annualized return 613.47%.
- Average Sharpe by ranking signal was led by `combo_rank` at 4.09.
- Average Sharpe by exit profile was led by `bull_put_2.0_8.0` at 6.03.


### Long-Only Spread Results
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPR_036 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_2.0_8.0; size=vol_inverse | 5.9586 | -29.9238 | 89.4794 | 615.7900 | 922 | True |
| SPR_048 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_2.0_8.0; size=vol_inverse | 5.7455 | -32.4391 | 87.8924 | 521.9150 | 892 | True |
| SPR_034 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_2.0_6.0; size=vol_inverse | 5.5154 | -32.6763 | 89.0456 | 554.4084 | 922 | True |
| SPR_030 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_1.0_8.0; size=vol_inverse | 5.5082 | -31.9368 | 88.1779 | 538.7090 | 922 | True |
| SPR_042 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_1.0_8.0; size=vol_inverse | 5.2535 | -34.5844 | 86.9955 | 450.4658 | 892 | True |
| SPR_035 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_2.0_8.0; size=equal | 5.2381 | -54.6004 | 88.4943 | 1403.4258 | 704 | True |
| SPR_046 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_2.0_6.0; size=vol_inverse | 5.2178 | -35.3574 | 87.5561 | 459.1098 | 892 | True |
| SPR_047 | rank=combo_rank; threshold=strict; concentration=spread_focus; exit=bull_put_2.0_8.0; size=equal | 5.1095 | -57.5064 | 87.9234 | 1235.3503 | 679 | True |
| SPR_032 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_2.0_4.0; size=vol_inverse | 5.0423 | -36.1337 | 87.7440 | 489.2309 | 922 | True |
| SPR_028 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_1.0_6.0; size=vol_inverse | 5.0359 | -35.0871 | 86.7679 | 474.2678 | 922 | True |

#### Holdout Check
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| SPR_036 | 2018_2024 | 5.7545 | -29.9238 | 589.7654 | 88.9851 | 808 |
| SPR_036 | 2025 | 7.4743 | -6.2105 | 782.9224 | 92.5000 | 120 |
| SPR_048 | 2018_2024 | 5.4890 | -32.4391 | 479.9935 | 86.8895 | 778 |
| SPR_048 | 2025 | 7.4743 | -6.2105 | 782.9224 | 92.5000 | 120 |
| SPR_034 | 2018_2024 | 5.3022 | -32.6763 | 527.6598 | 88.4901 | 808 |
| SPR_034 | 2025 | 7.0079 | -7.7797 | 714.8796 | 91.6667 | 120 |
| SPR_030 | 2018_2024 | 5.2934 | -31.9368 | 512.5909 | 87.6238 | 808 |
| SPR_030 | 2025 | 7.0624 | -7.1774 | 688.7224 | 91.6667 | 120 |
| SPR_042 | 2018_2024 | 4.9816 | -34.5844 | 410.5185 | 85.9897 | 778 |
| SPR_042 | 2025 | 7.0624 | -7.1774 | 688.7224 | 91.6667 | 120 |

- Best long-only spread results strategy: `SPR_036` with Sharpe 5.96, max drawdown -29.92%, and annualized return 615.79%.
- Average Sharpe by ranking signal was led by `combo_rank` at 3.63.
- Average Sharpe by exit profile was led by `bull_put_2.0_8.0` at 5.51.


### Short-Only Spread Results
| strategy_id | config | sharpe | max_drawdown | win_rate | annualized_return | trade_count | all_8_years_profitable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPR_048 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_2.0_8.0; size=vol_inverse | 5.5170 | -12.8804 | 92.8440 | 287.8341 | 545 | False |
| SPR_042 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_1.0_8.0; size=vol_inverse | 5.3076 | -14.0847 | 91.9266 | 267.3058 | 545 | False |
| SPR_046 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_2.0_6.0; size=vol_inverse | 5.3030 | -13.4223 | 92.4771 | 273.6336 | 545 | False |
| SPR_036 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_2.0_8.0; size=vol_inverse | 5.2348 | -16.0993 | 91.4557 | 330.3792 | 632 | True |
| SPR_035 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_2.0_8.0; size=equal | 5.0407 | -22.9577 | 92.3554 | 771.2730 | 484 | True |
| SPR_040 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_1.0_6.0; size=vol_inverse | 5.0186 | -15.1739 | 90.4587 | 249.2304 | 545 | False |
| SPR_044 | rank=combo_rank; threshold=strict; concentration=spread_balanced; exit=bull_put_2.0_4.0; size=vol_inverse | 4.9841 | -14.7117 | 91.0092 | 253.7663 | 545 | False |
| SPR_030 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_1.0_8.0; size=vol_inverse | 4.9805 | -16.2117 | 90.3481 | 301.8973 | 632 | True |
| SPR_034 | rank=combo_rank; threshold=balanced; concentration=spread_balanced; exit=bull_put_2.0_6.0; size=vol_inverse | 4.9694 | -16.1395 | 90.9810 | 309.4460 | 632 | True |
| SPR_029 | rank=combo_rank; threshold=balanced; concentration=spread_focus; exit=bull_put_1.0_8.0; size=equal | 4.8276 | -25.3571 | 91.5289 | 695.5490 | 484 | True |

#### Holdout Check
| strategy_id | period | sharpe | max_drawdown | annualized_return | win_rate | trade_count |
| --- | --- | --- | --- | --- | --- | --- |
| SPR_048 | 2018_2024 | 5.1613 | -12.8804 | 247.5515 | 92.3596 | 445 |
| SPR_048 | 2025 | 7.6999 | -3.1461 | 631.9119 | 95.0000 | 100 |
| SPR_042 | 2018_2024 | 4.9741 | -14.0847 | 230.7257 | 91.6854 | 445 |
| SPR_042 | 2025 | 7.3311 | -5.2266 | 574.2402 | 93.0000 | 100 |
| SPR_046 | 2018_2024 | 4.9641 | -13.4223 | 235.7970 | 92.1348 | 445 |
| SPR_046 | 2025 | 7.3664 | -4.7514 | 593.5398 | 94.0000 | 100 |
| SPR_036 | 2018_2024 | 4.7974 | -16.0993 | 269.8897 | 90.4483 | 513 |
| SPR_036 | 2025 | 7.5637 | -8.9450 | 878.6387 | 95.7627 | 118 |
| SPR_035 | 2018_2024 | 4.6359 | -22.9577 | 595.3732 | 91.1168 | 394 |
| SPR_035 | 2025 | 7.3362 | -9.8440 | 3121.6383 | 96.7033 | 91 |

- Best short-only spread results strategy: `SPR_048` with Sharpe 5.52, max drawdown -12.88%, and annualized return 287.83%.
- Average Sharpe by ranking signal was led by `combo_rank` at 3.41.
- Average Sharpe by exit profile was led by `bull_put_2.0_8.0` at 5.13.


## Recommended Strategy Set
- Combined stock portfolio: `STK_045`
- Long-only stock sleeve: `STK_063`
- Short-only stock sleeve: `STK_009`
- Combined options account: `OPT_013`
- Combined spreads account: `SPR_048`

## Independent Observations
- The raw return forecast is only moderately correlated with realized return; the ML scorer is more reliable as a ranking layer than a point-estimate engine.
- `predicted_mfe` is more useful for designing capped or target-driven exits than for estimating exact terminal returns.
- The short side is materially weaker than the long side at the raw signal level, so separate reporting is necessary even when combined portfolios are allowed.
- Robustness improves materially once earnings-overlap trades are removed, and concentration control matters as much as ranking choice.
