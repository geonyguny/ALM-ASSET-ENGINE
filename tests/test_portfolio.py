import numpy as np
from portfolio_opt import (weights_gmv, weights_msr, weights_target,
                           portfolio_stats, cov_from_corr)

# 입력값 (예시, 직접 쓰신 포트폴리오 값으로 교체 가능)
mu = np.array([0.06, 0.08, 0.03])   # 기대수익률
v  = np.array([0.15, 0.20, 0.05])   # 표준편차
R  = np.array([[1.0, 0.3, -0.1],
               [0.3, 1.0, 0.0],
               [-0.1, 0.0, 1.0]])

Sigma = cov_from_corr(v, R)

w_gmv = weights_gmv(mu, v, R)
w_msr = weights_msr(mu, v, R, r_f=0.02)
w_tgt = weights_target(mu, v, R, mu_target=0.07)

print("\n=== 효율 포트폴리오 결과 ===")
for name, w in [("GMV", w_gmv), ("MSR", w_msr), ("Target 7%", w_tgt)]:
    ret, vol, sh = portfolio_stats(w, mu, Sigma, r_f=0.02)
    print(f"{name:10s} | w={np.round(w,4)} | return={ret:.4f} | vol={vol:.4f} | sharpe={sh:.4f}")
