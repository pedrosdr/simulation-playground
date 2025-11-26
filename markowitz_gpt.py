#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def plot_markowitz_frontier(r, rf=None, n_points=200, reg=0.0, show=True):
    """
    Plota a fronteira eficiente de Markowitz (risky-only) a partir de uma matriz N×T de retornos.

    Parâmetros
    ----------
    r : array-like (N x T)
        N séries de retornos (linhas) e T períodos (colunas).
    rf : float ou None, opcional
        Taxa livre de risco por período (mesma unidade de r). Se fornecida, plota carteira tangente e CML.
    n_points : int, opcional
        Número de pontos na grade de retornos-alvo para traçar a fronteira risky-only.
    reg : float, opcional
        Regularização ridge para a matriz de covariâncias (adicionada à diagonal).
        Ex.: 0.0 (sem), 1e-6 (leve), etc.
    show : bool, opcional
        Se True, exibe a figura com matplotlib.

    Retorna
    -------
    out : dict
        {
          'mu': vetor de retornos esperados (N,),
          'Sigma': matriz de covariâncias (N x N),
          'w_gmv': pesos da carteira de variância mínima global,
          'mu_gmv': retorno da GMV,
          'sig_gmv': desvio-padrão da GMV,
          'm_grid': grid de retornos-alvo,
          'sig_grid': desvios-padrão correspondentes,
          'mask_efficient': máscara booleana do ramo eficiente (m >= mu_gmv),
          # Se rf:
          'w_tan': pesos carteira tangente,
          'mu_tan': retorno da tangente,
          'sig_tan': desvio-padrão da tangente,
          'sharpe_tan': Sharpe da tangente
        }
    """
    r = np.asarray(r)
    if r.ndim != 2:
        raise ValueError("r deve ser 2D (N x T).")
    N, T = r.shape
    if N < 2 or T < 2:
        raise ValueError("Forneça pelo menos 2 ativos e 2 períodos.")

    # Estatísticas
    mu = r.mean(axis=1)                 # (N,)
    Sigma = np.cov(r, bias=False)       # (N x N), rowvar=True (default)
    if reg and reg > 0:
        ridge = reg * (np.trace(Sigma) / N)
        Sigma = Sigma + ridge * np.eye(N)

    ones = np.ones(N)
    Sig_inv = np.linalg.pinv(Sigma)     # pseudo-inversa (mais estável)

    # Constantes clássicas
    A = ones @ Sig_inv @ ones
    B = ones @ Sig_inv @ mu
    C = mu   @ Sig_inv @ mu
    Delta = A*C - B**2
    if Delta <= 0:
        raise ValueError("Delta <= 0: verifique Sigma (singular?) e os dados. Tente aumentar reg.")

    # GMV
    w_gmv = (Sig_inv @ ones) / A
    mu_gmv = B / A
    sig_gmv = np.sqrt(1.0 / A)

    # Grade de retornos-alvo (risky-only)
    m_min = float(mu.min()*0.9)
    m_max = float(mu.max()*1.2)
    if not (m_max > m_min):
        m_min, m_max = float(mu.mean()*0.8), float(mu.mean()*1.2)
    m_grid = np.linspace(m_min, m_max, n_points)

    weights = []
    sig_grid = []
    for m in m_grid:
        w = Sig_inv @ ( ((C - B*m)/Delta)*ones + ((A*m - B)/Delta)*mu )
        weights.append(w)
        var = w @ Sigma @ w
        sig_grid.append(np.sqrt(max(var, 0.0)))
    weights = np.stack(weights, axis=0)
    sig_grid = np.asarray(sig_grid)
    mask_efficient = (m_grid >= mu_gmv)

    out = {
        "mu": mu, "Sigma": Sigma, "w_gmv": w_gmv,
        "mu_gmv": float(mu_gmv), "sig_gmv": float(sig_gmv),
        "m_grid": m_grid, "sig_grid": sig_grid,
        "mask_efficient": mask_efficient
    }

    # Carteira tangente (se rf fornecido)
    if rf is not None:
        k = Sig_inv @ (mu - rf*ones)
        den = ones @ k
        if abs(den) > 1e-12:
            w_tan = k / den
            mu_tan = float(mu @ w_tan)
            sig_tan = float(np.sqrt(w_tan @ Sigma @ w_tan))
            sharpe_tan = (mu_tan - rf) / sig_tan if sig_tan > 0 else np.nan
            out.update({
                "w_tan": w_tan, "mu_tan": mu_tan,
                "sig_tan": sig_tan, "sharpe_tan": sharpe_tan
            })
        else:
            out.update({"w_tan": None, "mu_tan": None, "sig_tan": None, "sharpe_tan": None})

    # Plot
    if show:
        plt.figure()
        plt.plot(sig_grid, m_grid, label="Fronteira (risky-only)")
        plt.plot(sig_grid[mask_efficient], m_grid[mask_efficient], linewidth=2, label="Ramo eficiente")
        plt.scatter([sig_gmv], [mu_gmv], label="GMV")
        if rf is not None and out.get("w_tan") is not None:
            plt.scatter([out["sig_tan"]], [out["mu_tan"]], label="Tangente (r_f)")
            xs = np.linspace(0.0, out["sig_tan"]*1.2, 50)
            ys = rf + out["sharpe_tan"] * xs
            plt.plot(xs, ys, linestyle="--", label="CML")
        plt.xlabel("Volatilidade (desvio-padrão por período)")
        plt.ylabel("Retorno esperado por período")
        plt.title("Fronteira eficiente de Markowitz (a partir de r: N x T)")
        plt.legend()
        plt.show()

    return out

#%%
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#%%
tickers = [
    'VALE3.SA', 'PETR4.SA', 'BBAS3.SA', 
    'TAEE11.SA', 'CMIG4.SA', 'CPFE3.SA',
    'ITSA4.SA'
]
r_demo = yf.download(
    tickers=tickers, 
    start='2020-01-01', 
    progress=False,
    interval='1mo'
)['Close'].dropna().to_numpy()
r_demo = np.log(r_demo)
r_demo = np.diff(r_demo, axis=0)

r_demo = r_demo.T

#%%
# rf na mesma unidade (ex.: 0.003 ~ 0,3% por período)
res = plot_markowitz_frontier(r_demo, rf=np.log1p(0.08)/12, n_points=250, reg=1e-4, show=True)

# %%
